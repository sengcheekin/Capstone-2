import torch
from torchvision import transforms
from train import netG
import gradio as gr
import copy

# load on-demand models (trained on 25, 50, 100, 150 epochs)
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

# set to evaluation mode as some layers behave differently during training and evaluation.
ondemand_25.eval()
ondemand_50.eval()
ondemand_100.eval()
ondemand_150.eval()

# load static models (trained on 25, 50, 100, 150 epochs)
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

# transform functions convert PIL image to tensor and vice versa
transform = transforms.Compose([transforms.ToTensor()])
tensor_to_image = transforms.ToPILImage()    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to perform inference on all on-demand models
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

# Function to perform inference on all static models
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

# Gradio interface
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
