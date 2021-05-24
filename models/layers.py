import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import math

from utils import torch_utils

def seq_and_vec(seq_len, vec):
    return vec.unsqueeze(1).repeat(1,seq_len,1)


