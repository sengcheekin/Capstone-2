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

# on-demand models
ondemand_25 = copy.deepcopy(netG)
checkpoint_25 = torch.load('checkpoints/checkpoint_ondemand_25.pth')
ondemand_25.load_state_dict(checkpoint_25['model_state_dict'])

ondemand_50 = copy.deepcopy(netG)
checkpoint_50 = torch.load('checkpoints/checkpoint_ondemand_50.pth')
ondemand_50.load_state_dict(checkpoint_50['model_state_dict'])

ondemand_100 = copy.deepcopy(netG)
checkpoint_100 = torch.load('checkpoints/checkpoint_ondemand_100.pth')
ondemand_100.load_state_dict(checkpoint_100['model_state_dict'])

ondemand_150 = copy.deepcopy(netG)
checkpoint_150 = torch.load('checkpoints/checkpoint_ondemand_final.pth')
ondemand_150.load_state_dict(checkpoint_150['model_state_dict'])

ondemand_25.eval()
ondemand_50.eval()
ondemand_100.eval()
ondemand_150.eval()

# static models
static_25 = copy.deepcopy(netG)
checkpoint_25 = torch.load('checkpoints/checkpoint_static_25.pth')
static_25.load_state_dict(checkpoint_25['model_state_dict'])

static_50 = copy.deepcopy(netG)
checkpoint_50 = torch.load('checkpoints/checkpoint_static_50.pth')
static_50.load_state_dict(checkpoint_50['model_state_dict'])

static_100 = copy.deepcopy(netG)
checkpoint_100 = torch.load('checkpoints/checkpoint_static_100.pth')
static_100.load_state_dict(checkpoint_100['model_state_dict'])

static_150 = copy.deepcopy(netG)
checkpoint_150 = torch.load('checkpoints/checkpoint_static_final.pth')
static_150.load_state_dict(checkpoint_150['model_state_dict'])

static_25.eval()
static_50.eval()
static_100.eval()
static_150.eval()

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
#############################################################################
# # test psnr calculation
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
# # psnr_scores = ds.calc_avg_psnr(val_clean_dir, val_hazy_dirs, netG, device)
# ds.redistribute([5,5,5,5,5], train_hazy_dirs, main_train_dir)
# if set(os.listdir("datasets/data/train/hazy/level1")) != set(os.listdir("datasets/data/train/main")):
#     print("not equal")
#     print( set(os.listdir("datasets/data/train/hazy/level1")) - set(os.listdir("datasets/data/train/main")))

# print("--- %s seconds ---" % (time.time() - start_time))
##############################################################################################
# Gradio interface
transform = transforms.Compose([transforms.ToTensor()])
tensor_to_image = transforms.ToPILImage()    

def inference_ondemand(img):
    img = transform(img)
    img = img.unsqueeze(0)
    models = [ondemand_25, ondemand_50, ondemand_100, ondemand_150]
    outputs_ondemand = []
    
    with torch.no_grad():
        for model in models:
            output = model(img.to(device))
            output = tensor_to_image(torch.squeeze(output.detach().cpu()))
            outputs_ondemand.append(output)
        
    return outputs_ondemand


def inference_static(img):
    img = transform(img)
    img = img.unsqueeze(0)
    models = [static_25, static_50, static_100, static_150]
    outputs_static = []
    
    with torch.no_grad():
        for model in models:
            output = model(img.to(device))
            output = tensor_to_image(torch.squeeze(output.detach().cpu()))
            outputs_static.append(output)
        
    return outputs_static

with gr.Blocks() as demo:
    with gr.Tab("On-Demand Learning Model"):
        gr.Markdown("## On-Demand Learning Model")
        with gr.Row():
            input_image = gr.Image(height=256, width=256)
        
        button_ondemand = gr.Button("Dehaze")

        with gr.Row():
            gr.Markdown("## 25 epochs")
            gr.Markdown("## 50 epochs")
            gr.Markdown("## 100 epochs")
            gr.Markdown("## 150 epochs")
            
        with gr.Row():
            output_ondemand_25 = gr.Image(height=256,width=256)
            output_ondemand_50 = gr.Image(height=256,width=256)
            output_ondemand_100 = gr.Image(height=256,width=256)
            output_ondemand_150 = gr.Image(height=256,width=256)


        button_ondemand.click(inference_ondemand, inputs = input_image, outputs = [output_ondemand_25, output_ondemand_50, output_ondemand_100, output_ondemand_150])

    with gr.Tab("Static Learning Model"):
            gr.Markdown("## Static Learning Model")
            with gr.Row():
                input_image = gr.Image(height=256, width=256)
            
            button_static = gr.Button("Dehaze")

            with gr.Row():
                gr.Markdown("## 25 epochs")
                gr.Markdown("## 50 epochs")
                gr.Markdown("## 100 epochs")
                gr.Markdown("## 150 epochs")
                
            with gr.Row():
                output_static_25 = gr.Image(height=256,width=256)
                output_static_50 = gr.Image(height=256,width=256)
                output_static_100 = gr.Image(height=256,width=256)
                output_static_150 = gr.Image(height=256,width=256)


            button_static.click(inference_static, inputs = input_image, outputs = [output_static_25, output_static_50, output_static_100, output_static_150])


if __name__ == "__main__":

    demo.launch()
