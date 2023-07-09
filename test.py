import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import dataset as ds
from train import netG
from matplotlib import pyplot as plt
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import os
import time
import gradio as gr
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
testloader_custom = ds.test_dataloader_custom

checkpoint = torch.load('checkpoints/checkpoint_ondemand_final.pth')
netG.load_state_dict(checkpoint['model_state_dict'])

netG_0 = copy.deepcopy(netG)
checkpoint_0 = torch.load('checkpoints/checkpoint_ondemand_0.pth')
netG_0.load_state_dict(checkpoint_0['model_state_dict'])

netG_50 = copy.deepcopy(netG)
checkpoint_50 = torch.load('checkpoints/checkpoint_ondemand_50.pth')
netG_50.load_state_dict(checkpoint_50['model_state_dict'])

netG_100 = copy.deepcopy(netG)
checkpoint_100 = torch.load('checkpoints/checkpoint_ondemand_100.pth')
netG_100.load_state_dict(checkpoint_100['model_state_dict'])

netG_150 = copy.deepcopy(netG)
checkpoint_150 = torch.load('checkpoints/checkpoint_ondemand_final.pth')
netG_150.load_state_dict(checkpoint_150['model_state_dict'])

netG_0.eval()
netG_50.eval()
netG_100.eval()
netG_150.eval()
# img_batch, hazy_batch = next(iter(testloader_custom))
# with torch.inference_mode():
#     output = netG(hazy_batch.to(device))
#     # ds.calc_avg_psnr(ds.val_clean_dir, ds.val_hazy_dirs, netG, device)
#     # ds.calc_avg_ssim(ds.val_clean_dir, ds.val_hazy_dirs, netG, device)

# output = output.detach().cpu()

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
# plt.tight_layout()

# # Show the plot
# plt.show()

# test psnr calculation
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = torch.load('checkpoints/checkpoint_ondemand_final.pth')
# netG.load_state_dict(checkpoint['model_state_dict'])
# netG.eval()

# val_clean_dir = "datasets/data/val/clean"
# val_hazy_dirs = [
#     "datasets/data/val/hazy/level1",
#     "datasets/data/val/hazy/level2",
#     "datasets/data/val/hazy/level3",
#     "datasets/data/val/hazy/level4",
#     "datasets/data/val/hazy/level5",
# ]
# train_hazy_dirs = [
#     "datasets/data/train/hazy/level1",
#     "datasets/data/train/hazy/level2",
#     "datasets/data/train/hazy/level3",
#     "datasets/data/train/hazy/level4",
#     "datasets/data/train/hazy/level5",
# ]
# main_train_dir = "datasets/data/train/main"
# start_time = time.time()
# for i in range(20):
# psnr_scores = ds.calc_avg_psnr(val_clean_dir, val_hazy_dirs, netG, device)
# ds.redistribute([5,5,5,5,5], train_hazy_dirs, main_train_dir)
# if set(os.listdir("datasets/data/train/hazy/level1")) != set(os.listdir("datasets/data/train/main")):
#     print("not equal")
#     print( set(os.listdir("datasets/data/train/hazy/level1")) - set(os.listdir("datasets/data/train/main")))

# print("--- %s seconds ---" % (time.time() - start_time))

# Gradio interface
transform = transforms.Compose([transforms.ToTensor()])
tensor_to_image = transforms.ToPILImage()    

def inference(img):
    img = transform(img)
    img = img.unsqueeze(0)

    with torch.inference_mode():
        output = netG_150(img.to(device))

    output = output.detach().cpu()
    output = torch.squeeze(output)
    output = tensor_to_image(output)
    return output

def inference_all(img):
    img = transform(img)
    img = img.unsqueeze(0)
    
    with torch.inference_mode():
        output_0 = netG_0(img.to(device))
        output_50 = netG_50(img.to(device))
        output_100 = netG_100(img.to(device))
        
    output_0 = output_0.detach().cpu()
    output_50 = output_50.detach().cpu()
    output_100 = output_100.detach().cpu()

    output_0 = torch.squeeze(output_0)
    output_50 = torch.squeeze(output_50)
    output_100 = torch.squeeze(output_100)

    output_0 = tensor_to_image(output_0)
    output_50 = tensor_to_image(output_50)
    output_100 = tensor_to_image(output_100)

    return output_0, output_50, output_100,




with gr.Blocks() as demo:
    gr.Markdown("## Image Dehazing")
    with gr.Row():
        input_image = gr.Image(height=256, width=256)
        output_image = gr.Image(height=256,width=256)

    button = gr.Button("Dehaze")
    button.click(inference, inputs = input_image, outputs = output_image)

    with gr.Row():
        gr.Markdown("## 0 epochs")
        gr.Markdown("## 50 epochs")
        gr.Markdown("## 100 epochs")
        
    with gr.Row():
        output_image_0 = gr.Image(height=256,width=256)
        output_image_50 = gr.Image(height=256,width=256)
        output_image_100 = gr.Image(height=256,width=256)


    button_all = gr.Button("Dehaze")
    button_all.click(inference_all, inputs = input_image, outputs = [output_image_0, output_image_50, output_image_100])

if __name__ == "__main__":

    demo.launch()
