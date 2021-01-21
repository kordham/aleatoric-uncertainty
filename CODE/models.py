import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace

import numpy as np

# -----------------------------------------------------------------------------


class LeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5):
        super(LeNet, self).__init__()

        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))

        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        self.drop  = nn.Dropout(p=0.5)
        
        

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x

# -----------------------------------------------------------------------------

class LeNet_KG(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5):
        super(LeNet_KG, self).__init__()

        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))

        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        self.fc4   = nn.Linear(84, out_chan)
        self.drop  = nn.Dropout(p=0.5)

        self.n_samp = 100
        self.dist   = Normal

        
    def sample_sigma(self,v,s,y):

        # calc sigma:
        sig2 = F.softplus(s)

        # calc_d:
        d = torch.empty((v.size()[0],v.size()[1],self.n_samp))#, device=device)
        yd = torch.empty((v.size()[0],self.n_samp), dtype=torch.long)#, device=device)
        for t in range(self.n_samp):
            eps = self.dist(torch.tensor([0.0]), torch.tensor([1.0])).sample(sample_shape=v.size())#.to(device=device)
            x = v + torch.sqrt(sig2)*torch.squeeze(eps)
            d[:,:,t] = x
            yd[:,t] = y

        return d, yd


    def forward(self, x):
      
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        v = self.fc3(x)
        s = self.fc4(x)

        return v,s

    
# -----------------------------------------------------------------------------
