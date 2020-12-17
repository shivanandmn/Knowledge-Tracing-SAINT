import torch
import numpy as np
from torch import nn
import copy

def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool)

def lt_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).to(dtype=torch.bool)

def pos_encode(seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0)

def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])