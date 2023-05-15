import torch


def load_model(model_path, gpu=False):
    model = torch.load(model_path)
    if gpu:
        model = model.to("cuda")
    return model
