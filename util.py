# This file contains utility functions that are supposed to be used in the main training file, train.py.
# The functions are supposed to emulate the deprecated functions in the source code.
# Most of which are used to create the channel-wise fully connected layer.
# However, the channel-wise fully connected layer was removed as it does not provide any significant improvement in the results.


# Do note that the code below cannot be used as it is, as it fails to produce the same results as the deprecated functions.
# Hence, this code was not used in the final implementation of the project, 
# and only serves as a reference and proof of effort spent on recreating the deprecated functions.

import torch
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        print(x.size())
        return x.reshape(self.shape)

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

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(16, 16) for i in range(512)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for linear in self.linears:
            x = linear(x)
        return x


