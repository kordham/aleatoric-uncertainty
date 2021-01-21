





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

from pruner import Pruner

import pandas as pd

def get_std(val_arr,num_bins):
    hist, bin_edges = np.histogram(val_arr,bins=num_bins,density=True)
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
        
def get_batch_sigma(model,train_loader):
    with torch.no_grad():
            model.eval()
            train_ale_unc = []; acc = 0
            batch_sigma = []
            for i, (x_train, y_train) in enumerate(train_loader):
                train_pred, train_std = model(x_train)
                
                for training_data_point in range(train_std.size()[0]):
                    batch_sigma.append(np.sqrt(F.softplus(train_std[training_data_point,train_pred.argmax(dim=1)[training_data_point].item()]).item()))
    return batch_sigma
        
def refine_learn_data(train_loader_in,batch_sigma,num_bins):
    train_loader = train_loader_in
    mu, sigma = get_std(batch_sigma,num_bins);
    counter = 0;
    for i, j  in enumerate(train_loader):
        for k in range(list(j[1].size())[0]):
            if(batch_sigma[counter +k]>mu + 6*sigma):
                print(counter+k)
        counter = counter + list(j[1].size())[0];
    return train_loader



def train(model, train_data, val_data, epochs, batch_size
          ,learning_rate,decay,step_size,gamma,num_bins
          ,prune_milestones,prune_sigmas,prune_gamma):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    pruner = Pruner(train_data, prune_milestones,prune_sigmas,optimizer,gamma = prune_gamma)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    # output_file = 'bs_'+str(batch_size)+'_lr_'+str(learning_rate)+'_wd_'+str(decay)+'.csv'
    # with open(output_file, 'w', newline="") as f_out:
    #     writer = csv.writer(f_out, delimiter=',')
    #     writer.writerow(["Epoch",'Train Loss','Val Loss','Val Acc'])    
    
    
    df = pd.DataFrame(index=list(range(epochs)),columns = ["trainacc","trainloss","valacc","valloss"])
    
    epoch_trainaccs, epoch_valaccs = [], []
    epoch_trainloss, epoch_valloss = [], []
    for epoch in range(epochs):  # loop over the dataset multiple times
        t1 = time.perf_counter()
        model.train()
        train_losses,  train_accs = [], []; acc = 0
        for batch, (x_train, y_train) in enumerate(train_loader):
            
            model.zero_grad()
            pred, std = model(x_train)
            d, yd = model.sample_sigma(pred, std,y_train)
            loss = F.cross_entropy(pred,y_train) + loss_l2(d,yd)
            
            loss.backward()
            optimizer.step()
            
            acc = (pred.argmax(dim=-1) == y_train).to(torch.float32).mean()
            train_accs.append(acc.mean().item())
            train_losses.append(loss.item())   
            
        with torch.no_grad():
            model.eval()
            val_losses, val_accs = [], []; acc = 0
            for i, (x_val, y_val) in enumerate(val_loader):
                val_pred, val_std = model(x_val)
                val_d, val_yd = model.sample_sigma(val_pred, val_std,y_val)
                loss =  F.cross_entropy(val_pred,y_val) + loss_l2(val_d,val_yd)
                acc = (val_pred.argmax(dim=-1) == y_val).to(torch.float32).mean()
                val_losses.append(loss.item())
                val_accs.append(acc.mean().item())
        
            train_loader  = pruner.prune_dataloaders(model, train_loader,batch_size, prune_gamma)
        
        scheduler.step()
            
        
        print('Decay: {}, Epoch: {}, Loss: {}, TAccuracy: {}, Ignore_list_Size: {}, LR: {}'.format(decay,epoch, np.mean(val_losses), np.mean(train_accs), len(pruner.ignore_list),optimizer.param_groups[0]['lr']))
        
        df.loc[epoch, 'trainacc'] = np.mean(train_accs)
        df.loc[epoch,'trainloss'] = np.mean(train_losses)
        df.loc[epoch,'valacc'] = np.mean(val_accs)
        df.loc[epoch,'valloss'] = np.mean(val_losses)
                                     
        
        epoch_trainaccs.append(np.mean(train_accs))
        epoch_valaccs.append(np.mean(val_accs))
        epoch_trainloss.append(np.mean(train_losses))
        epoch_valloss.append(np.mean(val_losses))
        _results = [time.perf_counter()-t1,epoch, np.mean(train_losses), np.mean(val_losses), np.mean(val_accs)]
        # with open(output_file, 'a', newline="") as f_out:
        #     writer = csv.writer(f_out, delimiter=',')
        #     writer.writerow(_results)
    return model, df