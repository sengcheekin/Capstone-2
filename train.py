# Code to train the dehazing model. Note that this train.py script is meant to train brand new models from scratch.
# To train the dehazing model from a checkpoint, use the resume_training.py script instead.

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import dataset as ds
from matplotlib import pyplot as plt
import time

# Hyperparameters
nef = 64             # number of encoder filters in first conv layer
ngf = 64             # number of generator filters in first conv layer
lr = 0.001           # initial learning rate for adam
model = "ondemand"   # model type: ondemand or static
epochs = 50          # number of epochs to train

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
main_train_dir = "datasets/data/train/main" # Specify the main training directory "main" for ondemand and "static" for static

# set the number of threads to 1 to avoid data loading error, and set the default tensor type to FloatTensor
# torch.set_num_threads(1)
torch.set_default_tensor_type("torch.FloatTensor")

# Import DataLoader
trainloader_custom = ds.train_dataloader_custom

# Initialize the variables/weights of the neural network
def weights_init(m):
    """Initialize the weights of the neural network based on if its Convolution or Batch Normalization
    Args:
        m: PyTorch nn module

    Returns:
        None
    """

    # if m is instance of Conv2d, initialize the weights of the convolutional layer using a normal distribution with mean 0 and standard deviation 0.02
    # if layer has bias, initialize it to 0
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# Generator Network
print("Initializing generator...")
# nn.Sequential: A sequential container that add modules to it in the order they are passed in the constructor.
netE = nn.Sequential(
    nn.Conv2d(3, nef, 4, 2, 1),
    nn.BatchNorm2d(nef),
    nn.LeakyReLU(0.2, True),

    nn.Conv2d(nef, nef*2, 4, 2, 1),
    nn.BatchNorm2d(nef*2),
    nn.LeakyReLU(0.2, True),

    nn.Conv2d(nef*2, nef*4, 4, 2, 1),
    nn.BatchNorm2d(nef*4),
    nn.LeakyReLU(0.2, True),

    nn.Conv2d(nef*4, nef*8, 4, 2, 1),
    nn.BatchNorm2d(nef*8),
    nn.LeakyReLU(0.2, True),
)

# The channel-wise fully connected layer here was removed as it does not provide any significant improvement in the results.

netG = nn.Sequential(
    netE,

    nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1),
    nn.BatchNorm2d(ngf*4),
    nn.ReLU(True),

    nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
    nn.BatchNorm2d(ngf*2),
    nn.ReLU(True),

    nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
    nn.BatchNorm2d(ngf),
    nn.ReLU(True),

    nn.ConvTranspose2d(ngf, 3, 4, 2, 1),
    nn.Tanh()
)

netG.apply(weights_init)

# Loss Metric (also called L2 loss)
criterion = nn.MSELoss()

# Setup optimizer
optimizer = optim.Adam(netG.parameters(), lr=lr)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG.to(device)

# Training
if __name__ == "__main__":
    print(f"Model selected: {model}, Epochs to train: {epochs}")
    print("Starting training...")
    train_time = time.time()
    epoch_arr = [] # For plotting purposes

    for epoch in range(epochs):
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
                }, f"checkpoints/checkpoint_{model}_{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}. File path: checkpoints/checkpoint_{model}_{epoch}.pth")

            # This section is to calculate the psnr from the validation dataset and redistribute the dataset accordingly, if the model is ondemand
            if model == "ondemand":
                netG.eval()
                psnr_scores = ds.calc_avg_psnr(val_clean_dir, val_hazy_dirs, netG, device)
                ds.redistribute(psnr_scores, train_hazy_dirs, main_train_dir)
                netG.train()
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}.., Loss: {epoch_loss:.4f}, Time: {time.time() - epoch_time}s ")

    print("Finished Training")
    print(f"Training time: {time.time() - train_time}s")

    # Save model and optimizer
    torch.save({
            'epoch': epoch,
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f"checkpoints/checkpoint_{model}_final.pth")
    
    
    # Plot loss
    plt.plot(epoch_arr)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


