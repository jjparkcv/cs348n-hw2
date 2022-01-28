import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn

from network import Network


def build_network(input_dim=3):
    net = Network(input_dim=input_dim)
    for k, v in net.named_parameters():
        if 'weight' in k:
            std = np.sqrt(2) / np.sqrt(v.shape[0])
            nn.init.normal_(v, 0.0, std)
        if 'bias' in k:
            nn.init.constant_(v, 0)
        if k == 'l_out.weight':
            std = np.sqrt(np.pi) / np.sqrt(v.shape[1])
            nn.init.constant_(v, std)
        if k == 'l_out.bias':
            nn.init.constant_(v, -1)            
    return net
    
def sample_fake(pts, noise=0.3):
    sampled = pts + torch.normal(0, 1, pts.shape) * noise
    return sampled
