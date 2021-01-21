import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
#from torchsummary import summary

import numpy as np
import PIL

from models import LeNet, LeNet_KG
from FRDEEP import FRDEEPN
from MiraBest import MiraBest_full, MBFRConfident
from training import train
import matplotlib.pyplot as plt
import csv
import os
import ast

import pandas as pd

from parameter_search import param_search

def output_images_with_class(folder,loader,model):
    
    
    
    model.eval()
    
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    
    with torch.no_grad():
        for batch, (x,y) in enumerate(loader):
             pred, s = model(x)
             c = pred.argmax(dim=1)
             var = F.softplus(s)
             for i in range(y.size()[0]):
                 imagepath = folder + "/batch{:01d}_img{:01d}_p{}_y{}_var{:.4e}.png".format(batch,i,c[i].item(),y[i].item(),var[i][c[i]].item())
                 plt.imsave(imagepath,x[i][0])


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


epochs        = 50  # number of training epochs
#valid_size    = 50    # number of samples for validation
batch_size    = 32    # number of samples per mini-batch
imsize        = 50    # image size
num_classes   = 2     # The number of output classes. FRI/FRII
learning_rate = 1e-3  # The speed of convergence
momentum      = 9e-1  # momentum for optimizer
decay         = 0.5e-2  # weight decay for regularisation
random_seed   = 42
step_size = 5
gamma = 0.8
num_bins = 30
prune_milestones = []
prune_sigmas = []
prune_gamma = 2

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Creating the data------------------------------------------------------------

transform = transforms.Compose([
    transforms.CenterCrop(imsize),
    transforms.RandomRotation([0,360],resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.0229,), (0.0957,))])

train_val_data = MBFRConfident('mirabest', train=True, download=True, transform=transform)


ntrainval = len(train_val_data)

TRAIN_PCT = 0.8
ntrain = int(ntrainval*TRAIN_PCT)

train_data,val_data = torch.utils.data.random_split(train_val_data,[ntrain,ntrainval-ntrain],generator=torch.Generator().manual_seed(random_seed))


#train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)



test_data = MBFRConfident('mirabest', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
ntest  = len(test_data)

#Saved model
outfile = "./mb_lenet_kg_20201110"
IS_TRAINED = True

# Initialise model-----------------------------------------------




    
model = LeNet_KG(in_chan=1, out_chan=num_classes, imsize=imsize, kernel_size=5)

params = {'epochs': [200],
          'batch_size' : [32],
          'lr' : [1e-3],
          'decay' : [1e-2],
          'gamma' : [0.8],
          'step_size' : [5],
          'num_bins' : [100],
          'prune_milestones' : [[50],[100],[150],[200]],
          'prune_sigmas' : [[5],[3],[2]],
          'prune_gamma' : [1e-3]
          }


results = param_search(model,params,train_data,val_data,test_data)

min_params = ast.literal_eval(min(results.keys(),key = lambda k: results[k]['num_errors']))

# if(IS_TRAINED==False):
#     model, df = train(
#         model,
#         train_data,
#         val_data,
#         epochs,  
#         batch_size,   
#         learning_rate,
#         decay,
#         step_size,
#         gamma,
#         num_bins,
#         prune_milestones,
#         prune_sigmas,
#         prune_gamma)
#     torch.save(model.state_dict(), outfile)
    


# else: #if the model is already trained, load the model from save file
#     model.load_state_dict(torch.load(outfile))

# model.eval()


    
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=False)
# with torch.no_grad():
#     test_outputs = []
#     std = []
#     for batch, (x_test,y_test) in enumerate(train_loader):
#           test_pred, test_s = model(x_test)
#           test_c = test_pred.argmax(dim=1)
#           test_std = torch.sqrt(F.softplus(test_s))
             
#     frac_unc = test_std/test_pred
    
#     plt.hist(test_pred.numpy().flatten(),20)
#     plt.hist(test_std.numpy().flatten(),20)
                 

    
# model.eval() #put the model into eval mode so it can be used for analysis




# #output_images_with_class('./testimages',test_loader,model)
# #output_images_with_class('./valimages',val_loader,model)






# train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=False)
# val_outputs, test_outputs = [], []
# with torch.no_grad():
    
#     for batch, (x_test,y_test) in enumerate(train_loader):
#           test_pred, test_s = model(x_test)
#           test_c = test_pred.argmax(dim=1)
#           test_var = F.softplus(test_s)
#           for i in range(y_test.size()[0]):
#               test_outputs.append([x_test[i][0],test_c[i].item(),y_test[i].item(),test_var[i][test_c[i]].item()])

# sorted_test_output = sorted(test_outputs,key = lambda i: i[3]) 

# s=0
# for it in sorted_test_output:
#     if it[1]==it[2]:
#         s+=1
        
# acc = s/len(sorted_test_output)

# print("test acc: ", acc)


