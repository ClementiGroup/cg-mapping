import numpy as np
import torch
from torch.nn import Parameter
from torch import nn, optim
import torch.nn.functional as F

class decoder_linear(torch.nn.Module):
    '''
    This is the linear decoder, which is a 3*N_CG by 3*N_atom matrix
    '''
    def __init__(self, in_dim, out_dim, device):
        super(decoder_linear, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.reset_parameters()
        self.device = device
        
    def reset_parameters(self):
        self.weight = Parameter(torch.rand(self.in_dim*3, self.out_dim*3))  
    def forward(self, xyz): 
        xyz = xyz.reshape((xyz.shape[0], 3*self.in_dim))
        return torch.matmul(xyz, self.weight.to(self.device)).reshape((xyz.shape[0], self.out_dim, 3))
    
class decoder_nonlinear_relu(torch.nn.Module):
    '''
    This is a 4-layer non-linear decoder with relu as activation function 
    '''
    def __init__(self, in_dim, out_dim, device):
        super(decoder_nonlinear_relu, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.latent_dim = out_dim * 3
        self.device = device
        
        self.input_layer = torch.nn.Linear(3*self.in_dim, self.latent_dim).to(self.device)
        self.latent_layer_1 = torch.nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
        self.latent_layer_2 = torch.nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
        self.latent_layer_3 = torch.nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
        self.output_layer = torch.nn.Linear(self.latent_dim, 3*self.out_dim).to(self.device)
        
    def forward(self, xyz): 
        xyz = xyz.reshape((xyz.shape[0], 3*self.in_dim))
        
        # Input layer
        xyz = torch.relu(self.input_layer(xyz))
        
        # Latent layer 1
        xyz = torch.relu(self.latent_layer_1(xyz))
        
        # Latent layer 2
        xyz = torch.relu(self.latent_layer_2(xyz))
        
        # Latent layer 3
        xyz = torch.relu(self.latent_layer_3(xyz))
        
        # Output layer
        xyz = self.output_layer(xyz)
        
        xyz = xyz.reshape((xyz.shape[0], self.out_dim, 3))
        return xyz