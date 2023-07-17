# Code to evaluate the model and calculate the average PSNR and SSIM

import torch
from torchvision import transforms
from train import netG
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import numpy as np
from PIL import Image

# standard images for testing
clean = Image.open("datasets/data/test/clean/berlin_000000_000019_leftImg8bit.jpg")

# put all level of haze in a list
hazy_list = [
        Image.open("datasets/data/test/hazy/level1/berlin_000000_000019_leftImg8bit_hazy.jpg"),
        Image.open("datasets/data/test/hazy/level2/berlin_000000_000019_leftImg8bit_hazy.jpg"),
        Image.open("datasets/data/test/hazy/level3/berlin_000000_000019_leftImg8bit_hazy.jpg"),
        Image.open("datasets/data/test/hazy/level4/berlin_000000_000019_leftImg8bit_hazy.jpg"),
        Image.open("datasets/data/test/hazy/level5/berlin_000000_000019_leftImg8bit_hazy.jpg")
        ]

# external images for testing (dense-haze dataset)
# clean = Image.open("datasets/dense-haze/clean/01_GT.jpg")
# hazy_list = [Image.open("datasets/dense-haze/hazy/01_hazy.jpg")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('checkpoints/checkpoint_ondemand_100.pth')
netG.load_state_dict(checkpoint['model_state_dict'])

transform = transforms.Compose([transforms.ToTensor()])
tensor_to_image = transforms.ToPILImage()

clean = transform(clean)
output_array = []

print("Epoch " + str(checkpoint['epoch']) + " loaded")
for i in range(len(hazy_list)):

    hazy = transform(hazy_list[i])

    hazy = hazy.unsqueeze(0)

    with torch.inference_mode():
        output = netG(hazy.to(device))

    output = output.detach().cpu()
    output = torch.squeeze(output)

    psnr = peak_signal_noise_ratio(clean, output)
    ssim = structural_similarity_index_measure(clean.unsqueeze(0), output.unsqueeze(0))

    output = tensor_to_image(output)

    # convert the image to array using numpy
    output = np.array(output)
    output_array.append(output)
    # output.show()

    print("PSNR of level " + str(i+1) + " haze: " + str(psnr))
    print("SSIM of level " + str(i+1) + " haze: " + str(ssim))

# concatenate all the images in the list
output_array = np.concatenate(output_array, axis=1)

# convert the array to image
output_array = Image.fromarray(output_array)

# show the image
output_array.show()





