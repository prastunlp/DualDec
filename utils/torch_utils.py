import torch
from torch import nn, optim
import os
# from torch.optim import Optimizer

def get_optimizer(name, parameters, lr, weight_decay):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat

def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var

def keep_partial_grad(grad, topk):
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad

def save(model, optimizer, opt, filename):
    params = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': opt
    }
    try:
        torch.save(params, filename)
    except BaseException:
        print("[ Warning: model saving failed. ]")

def load(model, optimizer, filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    if model is not None:
        model.load_state_dict(dump['model'])
    if optimizer is not None:
        optimizer.load_state_dict(dump['optimizer'])
    opt = dump['config']
    return model, optimizer, opt

def load_config(filename):
    try:
        print(os.path.isfile(filename))
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    return dump['config']


