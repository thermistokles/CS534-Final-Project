"""
Execution script for timms

Author: Ivan Klevanski

Notable References:
    https://www.kaggle.com/code/binhhuunguyen/pneumothorax-classification-training
    https://github.com/huggingface/pytorch-image-models/blob/main/train.py


"""
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn as skl
import timm
import torch.cuda
import albumentations
import albumentations.pytorch

from PIL import Image
from timm import utils
from timm.data import create_dataset
from timm.models import create_model
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Models
import timm.models.fastvit as fastvit
import fastervit.models.faster_vit as fastervit

# TODO: to be converted to command-line arguments
full_image_path = "..\\data\\siim-acr-pneumothorax\\"

# Machine-specific variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
num_epochs = 1  # 100
image_resize = 224

# Timm variables
trainDataSet = Dataset
trainDataLoader = DataLoader

validationDataSet = Dataset
validationDataLoader = DataLoader

testDataSet = Dataset
testDataLoader = DataLoader

# Results
best_model_weights = dict()


class STImageDataset(Dataset):
    """
    Sequential Tensor Image Dataset
    """

    def __init__(self, imageList, labelList, augmentation=None):
        self.images = imageList
        self.labels = labelList
        self.aug = augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        path = self.images[item]
        # Convert to raw byte array
        image = np.asarray(Image.open(path).convert("RGB"))
        # Format/adjust image
        if self.aug:
            image = self.aug(image=image)["image"]
        label = self.labels[item]
        return image, label


def env_setup():
    """
    Set up the application environment
    :return: (void)
    """
    # TODO: Finalize
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

    np.random.seed(42)
    utils.random_seed(seed, 0)
    torch.manual_seed(seed)


def data_preprocessing():
    """
    Load and preprocess data
    :return: (void)
    """
    masks_exp = full_image_path + "png_masks"
    raw_images_exp = full_image_path + "png_images"

    # Masks
    mfiles = []
    mpaths = []

    # Raw images
    files = []
    paths = []

    # Allocate mask arrays
    for dirname, _, filenames in os.walk(masks_exp):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            mpaths.append(path)
            mfiles.append(filename)

    # Allocate raw image arrays
    for dirname, _, filenames in os.walk(raw_images_exp):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            paths.append(path)
            files.append(filename)

    # Allocate format dataframes

    total_train_data = pd.read_csv("../data/siim-acr-pneumothorax/stage_1_train_images.csv")
    testImageData = pd.read_csv("../data/siim-acr-pneumothorax/stage_1_test_images.csv")

    total_train_data["path"] = total_train_data["new_filename"].apply(lambda x: os.path.join(raw_images_exp, x))
    total_train_data["mpath"] = total_train_data["new_filename"].apply(lambda x: os.path.join(masks_exp, x))

    testImageData["path"] = testImageData["new_filename"].apply(lambda x: os.path.join(raw_images_exp, x))
    testImageData["mpath"] = testImageData["new_filename"].apply(lambda x: os.path.join(masks_exp, x))

    # Symmetrize data types (same number of cases for having and not having pneumothorax)
    # TODO: Use Image Augmentation, 4000 images is probably not enough for a good model (maybe look into RandArgument)
    num_0 = len(total_train_data.loc[total_train_data["has_pneumo"] == 0])
    num_1 = len(total_train_data.loc[total_train_data["has_pneumo"] == 1])

    if num_0 > num_1:
        df = total_train_data.loc[total_train_data["has_pneumo"] == 0].iloc[(num_0 - num_1):]
        total_train_data = pd.merge(df, total_train_data.loc[total_train_data["has_pneumo"] == 1], how="outer")
        pass
    else:
        df = total_train_data.loc[total_train_data["has_pneumo"] == 1].iloc[(num_1 - num_0):]
        total_train_data = pd.merge(df, total_train_data.loc[total_train_data["has_pneumo"] == 0], how="outer")
        pass

    total_train_data = total_train_data.sample(random_state=seed, frac=1)
    train_images, val_images, train_labels, val_labels = skl.model_selection.train_test_split(
            total_train_data["path"].tolist(),
            total_train_data["has_pneumo"].tolist(),
            stratify=total_train_data["has_pneumo"].tolist(),
            train_size=0.9)

    # Initialize timm-specific fields
    global trainDataSet
    global trainDataLoader
    global validationDataSet
    global validationDataLoader
    global testDataSet
    global testDataLoader

    aug = albumentations.Compose([
        albumentations.Resize(width=image_resize, height=image_resize),
        albumentations.Normalize(max_pixel_value=255, always_apply=True),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    trainDataSet = STImageDataset(train_images, train_labels, aug)
    trainDataLoader = DataLoader(trainDataSet, shuffle=True)

    validationDataSet = STImageDataset(val_images, val_labels, aug)
    validationDataLoader = DataLoader(validationDataSet, shuffle=False)

    testDataSet = STImageDataset(testImageData["path"].tolist(), testImageData["has_pneumo"].tolist(), aug)
    testDataLoader = DataLoader(testDataSet, shuffle=False)


# TODO: Needs hyperparameter tuning, most of the implementation comes from first reference
def train_model(model=fastvit.fastvit_ma36(), specifier="fastvit"):
    if model is not None:
        m_best_weights = copy.deepcopy(model.state_dict())
        m_best_acc = 0

        model = model.to(device)
        # Potential hyperparameters
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fcn = nn.CrossEntropyLoss()
        dataloaders = dict({"train": trainDataLoader, "validation": validationDataLoader})

        for epoch in range(num_epochs):
            for phase in ["train", "validation"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0
                running_corrects = 0

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = loss_fcn(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)

                if phase == 'val' and epoch_acc >= m_best_acc:
                    m_best_acc = epoch_acc
                    m_best_weights = copy.deepcopy(model.state_dict())

        best_model_weights[specifier] = (m_best_weights, m_best_acc)
        pass
    pass


def main():
    env_setup()
    data_preprocessing()

    """
    Highest accuracy yielding/slowest:
    
    fastvit.fastvit_ma36()
    fastervit.faster_vit_6_224()
    
    Probable Hyperparameters:
    optimizer, drop rate, loss function, 
    
    """

    train_model()
    # train_model(fastervit.faster_vit_6_224(), "fastervit")
    pass


if __name__ == "__main__":
    main()
