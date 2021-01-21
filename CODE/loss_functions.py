import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

#kendall and gal loss function, TO DO
def loss_KG(d, yd):
     N = yd.size()[0]
     T = yd.size()[1]
     print((1/N) * ((1/T)* yd*F.softmax(d,dim=1)[:,1,:]).sum(dim=1).log().sum())
     return (1/N) * ((1/T)* yd*F.softmax(d,dim=1)[:,1,:]).sum(dim=1).log().sum()


#l2 loss function from Kochkina and Liakata paper
def loss_l2(d, yd):
        N = yd.size()[0]
        T = yd.size()[1]
        log_softmax_d = F.log_softmax(d,dim=1)
        
        S = torch.tensor([0.0],dtype=torch.float, requires_grad=True)
       
        for n in range(N):    
            S = S + log_softmax_d[n,yd[n,0]].sum()
        
        return -(1/N)*(1/T)*S