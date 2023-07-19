# Code to evaluate the model and calculate the average PSNR and SSIM

import torch
from torchvision import transforms
from train import netG
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import numpy as np
from PIL import Image
import os

# clean image for testing
clean = Image.open("datasets/data/test/clean/berlin_000000_000019_leftImg8bit.jpg")

# put all hazy images to test on into a list. Ensure that the images are the same scene, but with different levels of haze
hazy_list = [
        Image.open("datasets/data/test/hazy/level1/berlin_000000_000019_leftImg8bit_hazy.jpg"),
        Image.open("datasets/data/test/hazy/level2/berlin_000000_000019_leftImg8bit_hazy.jpg"),
        Image.open("datasets/data/test/hazy/level3/berlin_000000_000019_leftImg8bit_hazy.jpg"),
        Image.open("datasets/data/test/hazy/level4/berlin_000000_000019_leftImg8bit_hazy.jpg"),
        Image.open("datasets/data/test/hazy/level5/berlin_000000_000019_leftImg8bit_hazy.jpg")
        ]

# load the model
model = "static"    # set it to either "static" or "ondemand"
epoch = "final"     # set it to the epoch number of the model to be loaded (e.g. "100"). Range is from 0 to 145, in increments of 5.
                    # To use the final model, set it to "final"

# check if the checkpoint file exists before loading
if not os.path.exists(f'checkpoints/checkpoint_{model}_{epoch}.pth'):
    print("Checkpoint file does not exist. Please check if the checkpoint exists and try again.")
    exit(1)

checkpoint = torch.load(f'checkpoints/checkpoint_{model}_{epoch}.pth')
netG.load_state_dict(checkpoint['model_state_dict'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# functions to convert the image to tensor and vice versa
transform = transforms.Compose([transforms.ToTensor()])
tensor_to_image = transforms.ToPILImage()

clean = transform(clean)
output_array = []

print("Epoch " + epoch + " loaded")
print("Model: " + model)
for i in range(len(hazy_list)):

    hazy = transform(hazy_list[i])

    # add a dimension to the tensor to make it a batch of 1, so that it can be fed into the model
    hazy = hazy.unsqueeze(0)

    with torch.inference_mode():
        output = netG(hazy.to(device))

    output = output.detach().cpu()
    # remove the extra dimension
    output = torch.squeeze(output)

    psnr = peak_signal_noise_ratio(clean, output)
    ssim = structural_similarity_index_measure(clean.unsqueeze(0), output.unsqueeze(0))

    output = tensor_to_image(output)

    # convert the image to array using numpy, so that it can be concatenated
    output = np.array(output)
    output_array.append(output)

    print("PSNR of level " + str(i+1) + " haze: " + str(psnr))
    print("SSIM of level " + str(i+1) + " haze: " + str(ssim))

# concatenate all the images in the list
output_array = np.concatenate(output_array, axis=1)

# convert the array to image
output_array = Image.fromarray(output_array)

# show the image
output_array.show()





