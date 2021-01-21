
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
#from torchsummary import summary

import numpy as np
import PIL
import time
import matplotlib.pyplot as plt
import csv

from models import LeNet, LeNet_KG
from loss_functions import loss_l2
from torch.optim import lr_scheduler


class Pruner():
    
    def __init__(self, dataset, milestones, sigmas, optimizer, gamma = 1):
        
        if len(milestones) != len(sigmas):
            raise ValueError("Different numbers of milestones and sigmas given.")
        self.milestones = milestones
        self.sigmas = sigmas
        self.dataset = dataset
       
        self.sigmadict = dict([[i,j] for i,j in zip(self.milestones,self.sigmas)])
        
        self.step_number = 0
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        
        
        self.optimizer = optimizer
        self.re_learner = lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=gamma)
        
        self.ignore_list = []
        
        
    def get_std(self,val_arr,num_bins, plot=False):
        hist, bin_edges = np.histogram(val_arr,bins=num_bins,density=True)
        if(plot):
            plt.hist(val_arr,bins=num_bins)
            plt.show()
        HM = max(hist)/2
        index = np.argmax(hist)
        mu = bin_edges[index];
        limit = 0
        for i in range(index,len(hist)):
            if(hist[i]<HM):
                sigma = (bin_edges[i]-mu)/1.1775
                return mu, sigma
            
    def get_batch_sigma(self,model,train_loader):
        with torch.no_grad():
                model.eval()
                train_ale_unc = []; acc = 0
                batch_sigma = []
                for i, (x_train, y_train) in enumerate(train_loader):
                    train_pred, train_std = model(x_train)
                    
                    for training_data_point in range(train_std.size()[0]):
                        batch_sigma.append(np.sqrt(F.softplus(train_std[training_data_point,train_pred.argmax(dim=1)[training_data_point].item()]).item()) )
                    
        return batch_sigma
            
    
    def prune_dataloaders(self,model, train_loader,batch_size, lr):
        with torch.no_grad():
            model.eval()
            if(self.step_number in self.milestones):
                mu, sigma = self.get_std(self.get_batch_sigma(model,train_loader),50)
                self.ignore_list = []
                with torch.no_grad():
                    for i, (x,y) in enumerate(self.dataloader):
                        self.forward_add_ignore(model,x,mu+self.sigmadict[self.step_number]*sigma)
                train_loader, ignore_loader = self.create_dataloaders(self.dataset,batch_size)
                self.re_learner.step()
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
            self.step_number += 1
            return train_loader
        
        
    def forward_add_ignore(self, model, x, max_unc):
        with torch.no_grad():
            model.eval()
            v,s = model.forward(x)
            c = v.argmax(dim=1)
            std = torch.sqrt(F.softplus(s))
            for i in range(x.size()[0]):
                if std[i,c[i]] > max_unc and (i not in self.ignore_list):
                    self.ignore_list.append(i)
    
    def create_dataloaders(self,dataset,batch_size):
        indx = list(range(len(dataset)))
        
        for i in indx:
            if i in self.ignore_list:
                indx.remove(i)
        sub_ignore = torch.utils.data.Subset(dataset,self.ignore_list)
        sub_train = torch.utils.data.Subset(dataset,indx)
        
        ignore_loader = torch.utils.data.DataLoader(sub_ignore, batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(sub_train, batch_size, shuffle=True)
        
        return train_loader, ignore_loader
        
        
        