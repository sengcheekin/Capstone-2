import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import dataset as ds
import util
from matplotlib import pyplot as plt
from torchmetrics import PeakSignalNoiseRatio as psnr
import time

opt = {
    "batch_size": 100,  # number of samples to produce
    "load_size": 96,    # resize the loaded image to load size maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
    "fine_size": 64,    # size of random crops
    "nef": 64,          # number of encoder filters in first conv layer
    "ngf": 64,          # number of generator filters in first conv layer
    "nc": 3,            # number of channels in input
    "n_threads": 4,     # number of data loading threads to use
    "n_iter": 25,       # number of iterations at starting learning rate
    "lr": 0.0002,       # initial learning rate for adam
    "beta1": 0.5,       # momentum term of adam
    "n_train": float(
        "inf"
    ),                  # number of examples per epoch. float('inf') for full dataset
    "display": 1,       # display samples while training. 0 = false
    "display_id": 10,   # display window id.
    "display_iter": 50, # number of iterations after which display is updated
    "gpu": 1,           # gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    "name": "dehaze",   # name of the experiment you are running
    "manual_seed": 0,   # 0 means random seed
    "netG": "",         # '' means no pre-trained encoder-decoder net provided
    "base": 0,          # initial index base, 0 if training from scratch
    "level1": 20,       # number of training examples per batch for level 1 sub-task
    "level2": 20,       # number of training examples per batch for level 2 sub-task
    "level3": 20,       # number of training examples per batch for level 3 sub-task
    "level4": 20,       # number of training examples per batch for level 4 sub-task
    "level5": 20,       # number of training examples per batch for level 5 sub-task
}

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

# set seed
if opt["manual_seed"] == 0:
    opt["manual_seed"] = torch.randint(1, 10000, (1,)).item()

print("Seed: " + str(opt["manual_seed"]))
torch.manual_seed(opt["manual_seed"])
torch.set_num_threads(1)
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

    # if m is instance of Conv2d, initialize the weights of the convolutional layer with a normal distribution with mean 0 and standard deviation 0.02
    # if layer has bias, initialize it to 0
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


nc = opt["nc"]
ngf = opt["ngf"]
nef = opt["nef"]

# Generator Network
if opt["netG"] != "":
    netG = util.load_model(opt["netG"], opt["gpu"])
else:
    print("No pretrained generator provided. Initializing new generator.")
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
    
    # channel_wise: A sequential container that splits the input into chunks of the same size and applies a module to each chunk independently.
    # Determine if this is needed or not
    # channel_wise = nn.Sequential(
    #   util.Reshape((nef * 8, 16)), # Reshape acts as a substitute for nn.View in the source code
#       util.SplitTable( 1),
#       util.MyModule(),
#       util.JoinTable(2),
#   )
    
    netG = nn.Sequential(
      netE,
    #   channel_wise,
    #   util.Reshape((nef*8, 256, 4)), #   nn.Unflatten(nef*8, 4, 4),

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

    # Loss Metrics
    criterion = nn.MSELoss()

    # Setup Solver
    optimStateG = {
        "lr": opt["lr"],
        "beta1": opt["beta1"],
    }
    optimizer = optim.Adam(netG.parameters(), lr=opt["lr"]) # opt[beta1] not used    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG.to(device)

    # Training
    if __name__ == "__main__":
        print("Start training")
        train_time = time.time()
        epoch_arr = [] # For plotting purposes

        for epoch in range(50):
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


    
    

    # def show_images(images, nmax=64):
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     ax.set_xticks([]); ax.set_yticks([])
    #     ax.imshow(make_grid((images.detach().cpu()[:nmax]), nrow=8).permute(1, 2, 0))
    # def show_batch(dl, nmax=64):
    #     for images in dl:
    #         show_images(images, nmax)
    #         break

    # show_batch(output, 10)


    # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in netG.state_dict():
    #     print(param_tensor, "\t", netG.state_dict()[param_tensor].size())

    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

# # Testing
    # test_image = Image.open("datasets/data/train/hazy/aachen_000000_000019_leftImg8bit_hazy.jpg")
#     real_frame = Image.open("datasets/data/train/clean/aachen_000000_000019_leftImg8bit.jpg")
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#     ])
#     test_image = transform(test_image)
#     test_image = test_image.cuda()
#     test_image = test_image.unsqueeze_(0)

#     real_frame = transform(real_frame)
#     real_frame = real_frame.cuda()

#     print(test_image.shape)
#     print(real_frame.shape)

#     for epoch in range(100):
#         optimizer.zero_grad()

#         output = netG(test_image)
#         output = torch.squeeze(output)
#         print(output.shape)
#         loss = criterion(output, real_frame)
#         loss.backward()
#         optimizer.step()
    
#         print('Epoch {}, loss {}'.format(epoch, loss.item()))
    
    # output = output.detach().cpu()
    # print(type(output))
    # tensor_to_image = transforms.ToPILImage()
    # final_output = tensor_to_image(output)
    # final_output.show()



