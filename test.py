import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import dataset as ds
from train import netG
from matplotlib import pyplot as plt
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import peak_signal_noise_ratio
import os

# testloader_custom = ds.test_dataloader_custom
# checkpoint = torch.load('checkpoint.pth')
# netG.load_state_dict(checkpoint['model_state_dict'])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# img_batch, hazy_batch = next(iter(testloader_custom))
# netG.eval()
# with torch.inference_mode():
#     output = netG(hazy_batch.to(device))
#     ds.calc_avg_psnr(ds.test_clean_dir, ds.test_hazy_dir, netG, device)

# output = output.detach().cpu()

# test psnr calculation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('checkpoint.pth')
netG.load_state_dict(checkpoint['model_state_dict'])
netG.eval()
val_clean_dir = "datasets/data/val/clean"
val_hazy_dirs = [
    "datasets/data/val/hazy/level1",
    "datasets/data/val/hazy/level2",
    "datasets/data/val/hazy/level3",
    "datasets/data/val/hazy/level4",
    "datasets/data/val/hazy/level5",
]
ds.calc_avg_psnr(val_clean_dir, val_hazy_dirs, netG, device)

# # visualisation
# batch_size = hazy_batch.size(0)

# # Create a grid of subplots with 1 row and 2 columns
# fig, axes = plt.subplots(nrows=batch_size-5, ncols=2, figsize=(10, 10))

# # Iterate over each image in the batch
# for i in range(batch_size-5):
#     # Convert the tensor images to numpy arrays
#     hazy_image = hazy_batch[i].permute(1, 2, 0).cpu().numpy()
#     output_image = output[i].permute(1, 2, 0).cpu().numpy()

#     # Display hazy image in the left subplot
#     axes[i, 0].imshow(hazy_image, aspect='auto')
#     axes[i, 0].axis('off')

#     # Display output image in the right subplot
#     axes[i, 1].imshow(output_image, aspect='auto')
#     axes[i, 1].axis('off')

# # Adjust the spacing between subplots
# # plt.tight_layout()

# # Show the plot
# plt.show()
