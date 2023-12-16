print("Importing required libraries")

import os
import torch
import timm
from timm import utils
from PIL import Image
from tqdm import tqdm
import torch.cuda
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler

import torch.optim as optim
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

print("Import completed")

seed = 42
model = 'inception_next_tiny'
num_epochs = 20

print("Import inception_next_tiny")
#Creation of the model
model = timm.create_model(model, pretrained=True)

model.head.fc2 = nn.Linear(in_features=2304, out_features=1, bias=True)
print("Last layer of the model has been changed to output 1 class")

#Selecting GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The program will run on ", device)

#CUDA settings
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)

np.random.seed(seed)
utils.random_seed(seed, 0)
torch.manual_seed(seed)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Define a transformation to resize and normalize your images
# Define a transformation to resize and normalize your grayscale images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust to your model's input size
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Mean and std for grayscale images
])

#Reading the CSV Data for training
print("#Reading the CSV Data for training")
imageData=pd.read_csv('siim-acr-pneumothorax/stage_1_train_images.csv').reset_index(drop=True)

#Undersampling
# print("Performing undersampling on the minority class")
# rus = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = rus.fit_resample(imageData, imageData['has_pneumo'])

# print("Data balance:")
# print(y_resampled.value_counts())

# X_resampled = X_resampled['new_filename'].tolist()
# labels = y_resampled.to_numpy()

print("Image Preprocessing")
X = imageData['new_filename'].tolist()
labels = imageData['has_pneumo']

data_dir = 'siim-acr-pneumothorax/png_images'
image_filenames = os.listdir(data_dir)  # List of image file names in your data directory
image_paths = [os.path.join(data_dir, filename) for filename in image_filenames]

image_paths = []

for i in range(len(X)):
    file_name = 'siim-acr-pneumothorax/png_images/' + X[i]
    image_paths.append(file_name)

print("Number of images: ", len(image_paths))

#labels = imageData['has_pneumo'].to_numpy()
print("Number of labels: ", len(labels))

#Creating the training dataset
print("Creating the training dataset")
custom_dataset = CustomDataset(image_paths, labels, transform=transform)
train_loader = DataLoader(custom_dataset, batch_size=100, shuffle=True)

#Training the model with the dataset
print("Training...")
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    model.to(device)
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    # Calculate and print the average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

print("Training completed")

print("Testing...")
#Reading the CSV Data for testing
print("Reading the CSV Data for testing")
imageData=pd.read_csv('siim-acr-pneumothorax/stage_1_test_images.csv').reset_index(drop=True)

labels = imageData['has_pneumo'].to_numpy()
print("Number of labels: ", len(labels))

image_paths = []

for i in range(len(imageData)):
    file_name = 'siim-acr-pneumothorax/png_images/' + imageData['new_filename'][i]
    image_paths.append(file_name)

print("Number of images: ", len(image_paths))

custom_dataset = CustomDataset(image_paths, labels, transform=transform)

#Creating the dataset
print("Creating the dataset")
test_loader = DataLoader(custom_dataset, batch_size=100, shuffle=True)

#Testing the model
correct_predictions = 0
total_samples = 0
all_predicted = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        outputs = model(images)
        labels = labels.view(-1, 1)
        predicted = (outputs > 0.5).float()  # Assuming binary classification
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        #Move the predictions to CPU for processing SciKitLearn
        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predicted)
f1 = f1_score(all_labels, all_predicted)
conf_matrix = confusion_matrix(all_labels, all_predicted)
mcc = matthews_corrcoef(all_labels, all_predicted)

print("Testing completed")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"MCC Score: {mcc:.4f}")

print("Execution completed")