"""
Execution script for timms

Author: Ivan Klevanski

Notable References:
    https://www.kaggle.com/code/binhhuunguyen/pneumothorax-classification-training
    https://github.com/huggingface/pytorch-image-models/blob/main/train.py


"""
import argparse
import copy
import traceback

import fastervit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn as skl
import torch.nn as nn
import torch.cuda
import albumentations
import albumentations.pytorch

from PIL import Image
from timm import utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Models
from timm.models import *
from fastervit.models.faster_vit import *

# Parse Arguments
argParser = argparse.ArgumentParser()

argParser.add_argument("-m", "--model", type=str, default="fastvit_t8", help="Timm Model Specifier")
argParser.add_argument("-p", "--path", type=str, default="..\\data\\siim-acr-pneumothorax", help="Images path")
argParser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
argParser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch Size")
argParser.add_argument("-o", "--output", type=str, default="\\output", help="Output directory")
argParser.add_argument("-pm", "--pretrained_model", type=str, default="", help="Path to pretrained model")
args = argParser.parse_args()

full_image_path = args.path
model_name = args.model
num_epochs = args.epochs
batch_size = args.batch_size
output_path = str(str(os.getcwd()) + args.output)
pretrained_model = args.pretrained_model

# Machine-specific variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
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


class STMaskedImageDataset(Dataset):
    """
    Sequential Tensor Image Dataset
    """

    def __init__(self, image_list, mask_list, label_list, augmentation=None):
        self.images = image_list
        self.masks = mask_list
        self.labels = label_list
        self.aug = augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        path = self.images[item]
        mpath = self.masks[item]

        # Convert to raw byte array
        image = np.asarray(Image.open(path).convert("RGB"))
        mask = np.asarray(Image.open(mpath).convert("RGB"))

        # Add mask to image
        masked = np.where(mask == 0, torch.from_numpy(image), 0)

        # Format/adjust image
        if self.aug:
            augmented = self.aug(image=masked)
            out_img = augmented["image"]  # .permute(1, 2, 0) for debugging
        else:
            out_img = torch.from_numpy(masked)
        label = self.labels[item]
        return out_img, label


def env_setup():
    """
    Set up the application environment
    :return: (void)
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

    np.random.seed(seed)
    utils.random_seed(seed, 0)
    torch.manual_seed(seed)


def data_preprocessing():
    """
    Load and preprocess data
    :return: (void)
    """
    masks_exp = full_image_path + "/png_masks"
    raw_images_exp = full_image_path + "/png_images"

    # Allocate format dataframes

    total_train_data = pd.read_csv(full_image_path + "/stage_1_train_images.csv")
    test_image_data = pd.read_csv(full_image_path + "/stage_1_test_images.csv")

    total_train_data["path"] = total_train_data["new_filename"].apply(lambda x: os.path.join(raw_images_exp, x))
    total_train_data["mpath"] = total_train_data["new_filename"].apply(lambda x: os.path.join(masks_exp, x))

    test_image_data["path"] = test_image_data["new_filename"].apply(lambda x: os.path.join(raw_images_exp, x))
    test_image_data["mpath"] = test_image_data["new_filename"].apply(lambda x: os.path.join(masks_exp, x))

    # Symmetrize data types (same number of cases for having and not having pneumothorax)
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

    num_0 = len(test_image_data.loc[test_image_data["has_pneumo"] == 0])
    num_1 = len(test_image_data.loc[test_image_data["has_pneumo"] == 1])

    if num_0 > num_1:
        df = test_image_data.loc[test_image_data["has_pneumo"] == 0].iloc[(num_0 - num_1):]
        test_image_data = pd.merge(df, test_image_data.loc[test_image_data["has_pneumo"] == 1], how="outer")
        pass
    else:
        df = test_image_data.loc[test_image_data["has_pneumo"] == 1].iloc[(num_1 - num_0):]
        test_image_data = pd.merge(df, test_image_data.loc[test_image_data["has_pneumo"] == 0], how="outer")
        pass

    total_train_data = total_train_data.sample(random_state=seed, frac=1)
    test_image_data = test_image_data.sample(random_state=seed, frac=1)
    train_images, val_images, train_labels, val_labels = skl.model_selection.train_test_split(
        pd.DataFrame({"path": total_train_data["path"].tolist(),
                      "mpath": total_train_data["mpath"].tolist()}),
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

    trainDataSet = STMaskedImageDataset(train_images["path"].tolist(),
                                        train_images["mpath"].tolist(),
                                        train_labels, aug)
    trainDataLoader = DataLoader(trainDataSet, shuffle=True, batch_size=batch_size)

    validationDataSet = STMaskedImageDataset(val_images["path"].tolist(),
                                             val_images["mpath"].tolist(),
                                             val_labels, aug)
    validationDataLoader = DataLoader(validationDataSet, shuffle=False, batch_size=batch_size)

    testDataSet = STMaskedImageDataset(test_image_data["path"].tolist(),
                                       test_image_data["mpath"].tolist(),
                                       test_image_data["has_pneumo"].tolist(), aug)
    testDataLoader = DataLoader(testDataSet, shuffle=False, batch_size=batch_size)


def train_model(model=None, specifier=""):
    """
    Trains and validates the neural network model
    :param model: neural network to train/validate
    :param specifier: model name used for saving results
    :return: (void)
    """

    def train(data_loader):
        """
        Training phase for model
        :param data_loader: Associated data loader for phase
        :return: A tuple consisting of the Epoch Loss and the Epoch Accuracy
        """
        model.train()

        run_loss = 0
        correct_predictions = 0

        for image, label in tqdm(data_loader, ascii=True):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(image)
                loss = loss_fn(outputs, label)

                loss.backward()
                optimizer.step()

                _, predictions = torch.max(outputs, 1)

            run_loss += loss.item() * image.size(0)
            correct_predictions += torch.sum(predictions == label.data)

        # 0: Epoch Loss, 1: Epoch Accuracy
        return run_loss / len(data_loader.dataset), correct_predictions.double() / len(data_loader.dataset)

    def evaluate(data_loader):
        """
        Evaluation/validation phase for model
        :param data_loader: Associated data loader for phase
        :return: A tuple consisting of the Epoch Loss and the Epoch Accuracy
        """
        model.eval()

        run_loss = 0
        correct_predictions = 0

        for image, label in tqdm(data_loader, ascii=True):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                outputs = model(image)
                loss = loss_fn(outputs, label)

                _, predictions = torch.max(outputs, 1)

            run_loss += loss.item() * image.size(0)
            correct_predictions += torch.sum(predictions == label.data)

        # 0: Epoch Loss, 1: Epoch Accuracy
        return run_loss / len(data_loader.dataset), correct_predictions.double() / len(data_loader.dataset)

    if model is not None:
        try:
            print('-' * 20)
            print("Training: " + specifier)
            print('-' * 20 + '\n')

            m_best_weights = copy.deepcopy(model.state_dict())
            m_best_acc = 0

            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(num_epochs):

                print('-' * 20)
                print("Epoch: " + str(epoch + 1) + " out of " + str(num_epochs))
                print('-' * 20)

                epoch_loss, epoch_acc = train(trainDataLoader)
                print("\nEpoch Summary: {} Loss: {:.4f} Acc: {:.4f}".format("Train", epoch_loss, epoch_acc))

                epoch_loss, epoch_acc = evaluate(testDataLoader)
                print("\nEpoch Summary: {} Loss: {:.4f} Acc: {:.4f}".format("Train", epoch_loss, epoch_acc))

                if epoch_acc >= m_best_acc:
                    m_best_acc = epoch_acc
                    m_best_weights = copy.deepcopy(model.state_dict())

            model.load_state_dict(m_best_weights)
            best_model_weights[specifier] = (m_best_weights, m_best_acc)

            print("Best validation accuracy: {:.4f}\n".format(m_best_acc))
        except Exception:
            print("[Error]: %s training failed due to an exception, exiting...\n" % specifier)
            print("[Error]: Exception occurred during training")
            traceback.print_exc()
            exit(1)
        pass
    pass


# Most of the implementation comes from first reference
def test_model(model=None, specifier=""):
    """
    Test the given trained model for final evaluation
    :param model: neural network to test
    :param specifier: model name
    :return: (void)
    """
    if model is not None:
        try:

            print('-' * 20)
            print("Testing: " + specifier)
            print('-' * 20)

            model.eval()

            predictions = []
            ground_truths = []

            with torch.no_grad():
                for img, label in tqdm(testDataLoader, ascii=True):
                    output = torch.nn.functional.softmax(model(img.to(device)), dim=1)
                    _, index = torch.topk(output, k=1, dim=1)
                    predictions.append(index.flatten())
                    ground_truths.append(label)

            predictions = torch.cat(predictions)
            ground_truths = torch.cat(ground_truths)

            predictions = predictions.cpu().detach().numpy()
            ground_truths = np.array(ground_truths)

            print("\n%s accuracy: %.4f\n" % (specifier, skl.metrics.accuracy_score(ground_truths, predictions)))
            print("%s f1-score: %.4f\n" % (specifier, skl.metrics.f1_score(ground_truths, predictions)))
            print("%s MCC: %.4f\n" % (specifier, skl.metrics.matthews_corrcoef(ground_truths, predictions)))

            # Confusion matrix
            cm = skl.metrics.confusion_matrix(ground_truths, predictions, normalize="all")
            print("Confusion Matrix:")
            print(str(cm))

            # ROC Curve
            fpr, tpr, _ = skl.metrics.roc_curve(ground_truths, predictions)
            print("\n%s ROC AUC Score: %.4f\n" % (specifier, skl.metrics.roc_auc_score(ground_truths, predictions)))
            plt.plot(fpr, tpr)
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.savefig(f"{output_path}/{specifier}_roc.png")

        except:
            print("[Error]: Exception occurred during testing:\n")
            traceback.print_exc()
    pass


def main():
    # Check cuda availability
    cuda_info = "Cuda modules loaded." if torch.cuda.is_available() else "Cuda modules not loaded."

    print("[Info]: " + cuda_info + '\n')

    env_setup()
    data_preprocessing()

    """
    Probable Hyperparameters:
    batch size, learning rate, number of layers (just choose different variations of the neural network available) 
    """
    if "faster_vit" in model_name:
        model = faster_vit_0_224(depths=[2, 2, 4, 2], drop_path_rate=0.0, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU)
    else:
        model = create_model(model_name)

    if pretrained_model == "":

        train_model(model, model_name)

        # Save the model
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        torch.save(model.state_dict(), f"{output_path}/{model_name}.pth")
    else:
        weights = torch.load(str(str(os.getcwd()) + pretrained_model))
        model.load_state_dict(weights)
        model = model.to(device)

    # Test the model
    test_model(model, model_name)
    pass


if __name__ == "__main__":
    main()
