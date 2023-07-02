
# This file is to continue training the model from a checkpoint
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import dataset as ds
import util
from matplotlib import pyplot as plt
from torchmetrics import PeakSignalNoiseRatio as psnr
import time
from train import netG, optimizer

val_clean_dir = "datasets/data/val/clean"
val_hazy_dirs = [
    "datasets/data/val/hazy/level1",
    "datasets/data/val/hazy/level2",
    "datasets/data/val/hazy/level3",
    "datasets/data/val/hazy/level4",
    "datasets/data/val/hazy/level5",
]
train_hazy_dirs = [
    "datasets/data/train/hazy/level1",
    "datasets/data/train/hazy/level2",
    "datasets/data/train/hazy/level3",
    "datasets/data/train/hazy/level4",
    "datasets/data/train/hazy/level5",
]
main_train_dir = "datasets/data/train/main"

# Import DataLoader
trainloader_custom = ds.train_dataloader_custom

checkpoint = torch.load('checkpoints/checkpoint_ondemand_45.pth')
netG.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1  # Specify the epoch to continue training from

# Loss Metrics
criterion = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG.to(device)

netG.train()

    # Training
if __name__ == "__main__":
    print("Start training")
    train_time = time.time()
    epoch_arr = [] # For plotting purposes

    for epoch in range(start_epoch, 150):
        epoch_time = time.time()
        running_loss = 0.0

        for clean, hazy in trainloader_custom:
            clean = clean.to(device)
            hazy = hazy.to(device)

            optimizer.zero_grad()

            output = netG(hazy)

            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            # Multiply by the first dimension of the input tensor (batch) to scale the loss value to the batch size
            running_loss += loss.item() * clean.size(0)
            
        
        epoch_loss = running_loss / len(trainloader_custom.dataset)
        epoch_arr.append(epoch_loss)

        # Save and evaluate performance on validation dataset every 5 epochs and redistribute the dataset accordingly
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': netG.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f"checkpoints/checkpoint_ondemand_{epoch}.pth")

            # This section is to calculate the psnr from the validation dataset and redistribute the dataset accordingly
            netG.eval()
            psnr_scores = ds.calc_avg_psnr(val_clean_dir, val_hazy_dirs, netG, device)
            ds.redistribute(psnr_scores, train_hazy_dirs, main_train_dir)
            netG.train()
        
        # Print progress
        print(f"Epoch {epoch+1}/{50}.., Loss: {epoch_loss:.4f}, Time: {time.time() - epoch_time}s ")

    print("Finished Training")
    print(f"Training time: {time.time() - train_time}s")

    # Save model and optimizer
    PATH = "checkpoints/checkpoint_ondemand_final.pth"
    torch.save({
            'epoch': epoch,
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)
    
    # Plot loss
    plt.plot(epoch_arr)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
