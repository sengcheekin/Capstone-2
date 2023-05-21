import torch
import torch.nn as nn


def load_model(model_path, gpu=False):
    model = torch.load(model_path)
    if gpu:
        model = model.to("cuda")
    return model

# Contains replacement functions for deprecated functions in the source code
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(self.shape)

# TODO: Determine if the argument dim is needed or not in the forward function of SplitTable and JoinTable.
#  Currently the dim is initialized as part of the class
class SplitTable(nn.Module):
    def __init__(self, dim):
        super(SplitTable, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.split(x.size(self.dim), self.dim) # split input tensors into sub-tensors along the dimension dim
    
class JoinTable(nn.Module):
    def __init__(self, dim):
        super(JoinTable, self).__init__()
        self.dim = dim

    def forward(self, x):
        y = torch.cat(x, self.dim)
        return y
