import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# from haze_synthesize import gen_haze

# dir_path = (
#     "D:/Documents/Semester 9/Capstone 2/Code/Capstone-2/datasets/data/train/test/clean"
# )

# data_transform = transforms.Compose(
#     [
#         transforms.Resize((64, 64)),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.ToTensor(),
#     ]
# )


# def plot_transformed_images(image_paths, transform, n=3, seed=42):
#     random.seed(seed)
#     random_image_paths = random.sample(image_paths, k=n)
#     for image_path in random_image_paths:
#         with Image.open(dir_path + "/" + image_path) as f:
#             fig, ax = plt.subplots(1, 2)
#             ax[0].imshow(f)
#             ax[0].set_title(f"Original \nSize: {f.size}")
#             ax[0].axis("off")

#             # Transform and plot image
#             # Note: permute() will change shape of image to suit matplotlib
#             # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
#             transformed_image = transform(f).permute(1, 2, 0)
#             ax[1].imshow(transformed_image)
#             ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
#             ax[1].axis("off")

#             fig.suptitle(f"Class: oui", fontsize=16)
#             # fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
#             plt.show()


# plot_transformed_images(os.listdir(dir_path), transform=data_transform, n=3)
# img = cv.imread("datasets/data/train/" + "aachen_000000_000019_leftImg8bit.jpg")
# plt.imshow(img)
# plt.show()

# dataset = datasets.ImageFolder(root="datasets/data", transform=data_transform)
# for i, (image, label) in enumerate(dataset):
#     print(f"Image {i} belongs to class {label}")

# print([entry.name for entry in list(os.scandir(dir_path))])

# Hyperparameters
train_dir = "datasets/data/train/clean"
hazy_dir = "datasets/data/train/hazy"
beta_range = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]
train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)


def get_classes(directory):
    classes = [
        dir.name for dir in os.scandir(os.path.dirname(directory)) if dir.is_dir()
    ]
    class_idx = {classes: i for i, classes in enumerate(classes)}

    return classes, class_idx


class HazyDataset(Dataset):
    def __init__(self, img_dir, hazy_dir, transform=None):
        self.img_dir = img_dir
        self.hazy_dir = hazy_dir
        self.transform = transform
        self.classes, self.class_idx = get_classes(img_dir)

    def __len__(self):
        print("img_dir:" + str(len(os.listdir(self.img_dir))), "hazy_dir:" + str(len(os.listdir(self.hazy_dir))))
        return len(os.listdir(self.img_dir))

    def __getitem__(self, index):
        # TODO: Determine if seperate directories for different levels of haze is needed
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[index])
        hazy_path = os.path.join(self.hazy_dir, os.listdir(self.hazy_dir)[index])
        print(index)
        img = Image.open(img_path)
        hazy = Image.open(hazy_path)

        if self.transform:
            return self.transform(img), self.transform(hazy)

        return img, hazy


train_data_custom = HazyDataset(
    img_dir=train_dir, hazy_dir=hazy_dir, transform=train_transform
)

train_dataloader_custom = DataLoader(train_data_custom, batch_size=100, num_workers=0 ,shuffle=True)

def display_random_images(dataset, n=3, seed=42):
    random.seed(seed)
    random_idx = random.sample(range(len(dataset)), k=n)
    for idx in random_idx:
        img, hazy = dataset[idx]
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img.permute(1,2,0))
        ax[0].set_title(f"Original \nSize: {img.shape}")
        ax[0].axis("off")

        # Transform and plot image
        # Note: permute() will change shape of image to suit matplotlib
        # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
        ax[1].imshow(hazy.permute(1,2,0))
        ax[1].set_title(f"Hazy \nSize: {hazy.shape}")
        ax[1].axis("off")

        fig.suptitle(f"Class: oui", fontsize=16)
        # fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
        plt.show()

# display_random_images(train_data_custom, n=3)
# train_data = datasets.ImageFolder(
#     root=os.path.dirname(train_dir), transform=train_transform, target_transform=None
# )

# test_data = datasets.ImageFolder(
#     root=os.path.dirname(hazy_dir), transform=test_transform
# )


# train_dataloader = (
#     DataLoader(dataset=train_data, batch_size=1, num_workers=8, shuffle=True),
# )
# test_dataloader = (
#     DataLoader(dataset=test_data, batch_size=1, num_workers=8, shuffle=True),
# )

# plot_transformed_images(os.listdir(train_dir), transform=train_transform, n=3)
# print(os.listdir(train_dir))
