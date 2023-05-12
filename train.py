import torch
import torch.nn as nn
import torch.optim as optim
import cv2 as cv
import numpy as np
import os

opt = {
    "batch_size": 100,  # number of samples to produce
    "load_size": 96,  # resize the loaded image to load size maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
    "fine_size": 64,  # size of random crops
    "nef": 64,  # number of encoder filters in first conv layer
    "ngf": 64,  # number of generator filters in first conv layer
    "nc": 3,  # number of channels in input
    "n_threads": 4,  # number of data loading threads to use
    "n_iter": 25,  # number of iterations at starting learning rate
    "lr": 0.0002,  # initial learning rate for adam
    "beta1": 0.5,  # momentum term of adam
    "n_train": float(
        "inf"
    ),  # number of examples per epoch. float('inf') for full dataset
    "display": 1,  # display samples while training. 0 = false
    "display_id": 10,  # display window id.
    "display_iter": 50,  # number of iterations after which display is updated
    "gpu": 1,  # gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    "name": "dehaze",  # name of the experiment you are running
    "manual_seed": 0,  # 0 means random seed
    "netG": "",  # '' means no pre-trained encoder-decoder net provided
    "base": 0,  # initial index base, 0 if training from scratch
    "level1": 20,  # number of training examples per batch for level 1 sub-task
    "level2": 20,  # number of training examples per batch for level 2 sub-task
    "level3": 20,  # number of training examples per batch for level 3 sub-task
    "level4": 20,  # number of training examples per batch for level 4 sub-task
    "level5": 20,  # number of training examples per batch for level 5 sub-task
}

# set seed
if opt["manual_seed"] == 0:
    opt["manual_seed"] = torch.random(1, 10000)

print("Seed: " + str(opt["manual_seed"]))
torch.manual_seed(opt["manual_seed"])
torch.set_num_threads(1)
torch.set_default_tensor_type("torch.FloatTensor")
