import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import numpy as np
import os
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
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
hazy_dir = "datasets/data/train/main"

test_clean_dir = "datasets\data/test\clean"
test_hazy_dir = "datasets/data/test/hazy/main"
test_hazy_dirs = [
    "datasets\data/test\hazy\level1",
    "datasets\data/test\hazy\level2",
    "datasets\data/test\hazy\level3",
    "datasets\data/test\hazy\level4",
    "datasets\data/test\hazy\level5",
]

val_clean_dir = "datasets/data/val/clean"
val_hazy_dirs = [
    "datasets/data/val/hazy/level1",
    "datasets/data/val/hazy/level2",
    "datasets/data/val/hazy/level3",
    "datasets/data/val/hazy/level4",
    "datasets/data/val/hazy/level5",
]

beta_range = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

def get_classes(directory):
    classes = [
        dir.name for dir in os.scandir(os.path.dirname(directory)) if dir.is_dir()
    ]
    class_idx = {classes: i for i, classes in enumerate(classes)}

    return classes, class_idx

# Takes in an array of scores, and redistribute the training batches inversely proportional to the scores
def redistribute(psnr_scores, sub_directories, main_directory):
    total_psnr = sum(psnr_scores)

    # Calculate the proportion inversely proportional to the psnr scores
    proportions = [total_psnr/psnr_score for psnr_score in psnr_scores]

    total_proportion = sum(proportions)
    inverse_proportions = [proportion/total_proportion for proportion in proportions]

    # Calculate number of samples to be taken from each dataset, inversely proportional to the scores
    num_samples = [int(inverse * len(os.listdir(directory))) for inverse, directory in zip(inverse_proportions, sub_directories)]

    # Calculate remaining samples to be distributed randomly
    remaining_samples = len(os.listdir(sub_directories[0])) - sum(num_samples)

    # Distribute remaining samples randomly
    random_indices = np.random.choice(5, size=remaining_samples, replace=True)
    for index in random_indices:
        num_samples[index] += 1

    # Copy files over from each hazy level to the main training directory according to the number of samples
    available_files = os.listdir(sub_directories[0])
    available_idx = [i for i in range(len(available_files))]
    for num_sample, sub_directory in zip(num_samples, sub_directories):
        
        # Select random files from the sub directory based on available indices
        random_idx = random.sample(available_idx, num_sample)

        # Get the files from the random indices
        random_files = [available_files[idx] for idx in random_idx]

        random_files = random.sample(os.listdir(sub_directory), num_sample)
        
        available_idx = list(set(available_idx) - set(random_idx))
        
        # Copy the selected files to the central file directory
        for file in random_files:
            file_path = os.path.join(sub_directory, file)
            shutil.copy2(file_path, main_directory)
        
    print(f"Number of samples taken: {list(zip(sub_directories, num_samples))}")

def calc_avg_psnr(clean_dir, hazy_dirs, model, device):

    psnr_levels = []

    for hazy_dir in hazy_dirs:

        val_dataset_custom = HazyDataset(clean_dir, hazy_dir, transform=transform)
        val_dataloader_custom = DataLoader(val_dataset_custom, batch_size=10, num_workers=0, shuffle=False)

        total_psnr = 0

        clean_img, hazy_img = next(iter(val_dataloader_custom))
        clean_img = clean_img.to(device)
        hazy_img = hazy_img.to(device)

        with torch.inference_mode():
            output = model(hazy_img)
        
        psnr = peak_signal_noise_ratio(output, clean_img)
        total_psnr += psnr

        avg_psnr = total_psnr / len(val_dataloader_custom)
        psnr_levels.append(avg_psnr)

    print("Psnr levels: ", [psnr.item() for psnr in psnr_levels])
    return [psnr.item() for psnr in psnr_levels]

#TODO: Implement function to calculate SSIM scores (for performance evaluation) 
def calc_avg_ssim(clean_dir, hazy_dirs, model, device):
    
        ssim_levels = []
    
        for hazy_dir in hazy_dirs:
    
            val_dataset_custom = HazyDataset(clean_dir, hazy_dir, transform=transform)
            val_dataloader_custom = DataLoader(val_dataset_custom, batch_size=10, num_workers=0, shuffle=False)
    
            total_ssim = 0
    
            clean_img, hazy_img = next(iter(val_dataloader_custom))
            clean_img = clean_img.to(device)
            hazy_img = hazy_img.to(device)
    
            with torch.inference_mode():
                output = model(hazy_img)
            
            ssim = structural_similarity_index_measure(output, clean_img)
            total_ssim += ssim
    
            avg_ssim = total_ssim / len(val_dataloader_custom)
            ssim_levels.append(avg_ssim)
    
        print("SSIM levels: ", [ssim.item() for ssim in ssim_levels])
        return [ssim.item() for ssim in ssim_levels]

class HazyDataset(Dataset):
    def __init__(self, img_dir, hazy_dir, transform=None):
        self.img_dir = img_dir
        self.hazy_dir = hazy_dir
        self.transform = transform
        self.classes, self.class_idx = get_classes(img_dir) # Not sure if get_classes is needed or not

    def __len__(self):
        # print("img_dir:" + str(len(os.listdir(self.img_dir))), "hazy_dir:" + str(len(os.listdir(self.hazy_dir))))
        return len(os.listdir(self.img_dir))

    def __getitem__(self, index):
        # TODO: Determine if seperate directories for different levels of haze is needed
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[index])
        hazy_path = os.path.join(self.hazy_dir, os.listdir(self.hazy_dir)[index])
        img = Image.open(img_path)
        hazy = Image.open(hazy_path)

        if self.transform:
            return self.transform(img), self.transform(hazy)

        return img, hazy
    

train_dataset_custom = HazyDataset(
    img_dir=train_dir, hazy_dir=hazy_dir, transform=transform
)
test_dataset_custom = HazyDataset(
    img_dir=test_clean_dir, hazy_dir=test_hazy_dir, transform=transform
)

train_dataloader_custom = DataLoader(train_dataset_custom, batch_size=10, num_workers=0 ,shuffle=True)
test_dataloader_custom = DataLoader(test_dataset_custom, batch_size=10, num_workers=0 ,shuffle=True)


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

# testing
