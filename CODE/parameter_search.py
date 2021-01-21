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
from training import train


from pruner import Pruner

import pandas as pd

import itertools

params = {'epochs': [200],
          'batch_size' : [32],
          'lr' : [1e-3],
          'decay' : [1e-2],
          'gamma' : [0.8],
          'step_size' : [5],
          'num_bins' : [100],
          'prune_milestones' : [[50]],#[100],[200]],
          'prune_sigmas' : [[5]],#[3],[2]],
          'prune_gamma' : [1e-3]
          }




def param_search(model,params, train_data, val_data, test_data):
    
    
    start_state = model.state_dict()
    
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)
        
    
    keys, values = zip(*params.items())
    param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    
    results = {}
    for p in param_dicts:
        print(p)
        model.load_state_dict(start_state)
        model, df = train(
            model,
            train_data,
            val_data,
            p['prune_milestones'][-1]+100,  
            p['batch_size'],
            p['lr'],
            p['decay'],
            p['step_size'],
            p['gamma'],
            p['num_bins'],
            p['prune_milestones'],
            p['prune_sigmas'],
            p['prune_gamma'])
        
        model.eval()
        
        with torch.no_grad():
            num_errors = 0
            for batch, (x,y) in enumerate(test_loader):
                pred, s  = model(x)
                c = pred.argmax(dim=1)
                num_errors += (c!=y).sum().item()
            
            results[str(p)] = {'model': model, 'df': df ,'num_errors' : num_errors}
            
    return results
    
    