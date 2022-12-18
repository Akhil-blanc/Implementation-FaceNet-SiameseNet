from models import *
from helpers import *
from loss import *

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, BatchSampler
import torchvision.utils
import torch
from torch import optim

from torchsummary import summary

#checking if cuda/gpu is available to use
cuda = torch.cuda.is_available()

data = datasets.CelebA(root = "./data", download = True)

# eps = 1e-8 

def create_model(network):
    if network.lower() == 'nn1':
        return nn1()
    elif network.lower() == 'nn2':
        return nn2()
    elif network.lower() == 'nn3':
        return nn3()
    elif network.lower() == 'nn4':
        return nn4()
    elif network.lower() == 'nns1':
        return nns1()
    else:
        raise ValueError("Architecture not defined.")

def train_all():
    for i in ['nn1', 'nn2', 'nn3', 'nn4', 'nns1']:
        model = create_model(i)

        #using cuda/gpu if available
        if cuda:
            model.cuda()
            
        #defining Adagrad optimizer 
        optimizer = optim.Adagrad(model.parameters(), lr = 0.005 )
        
        epochs = 1
        
        for j in range(epochs):
            counter = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                target = target if len(target) > 0 else None
                if not type(data) in (tuple, list):
                    data = (data,)
                if cuda:
                    data = tuple(d.cuda() for d in data)
                    if target is not None:
                        target = target.cuda()
                        
                embeddings = model(*data)
                optimizer.zero_grad()
                currloss = batch_hard_triplet_loss(target, embeddings, 0.2)
                currloss.backward()
                optimizer.step()
                counter+=1
                print(epochs," ",counter,"",currloss.item())
                
            summary(model,(3,220,220))
                
            torch.save(model.state_dict(), "FaceNet/saved_models/" + i + ".pkl")

print(1)
train_data = torchvision.datasets.CelebA(root = "./data", split = "train",target_type='identity', download = True, transform = transforms.Compose([transforms.Resize((220, 220)), transforms.PILToTensor(), transforms.ConvertImageDtype(dtype = torch.float32)]))
train_batch_sampler = BalancedBatchSampler(train_data, n_classes=2, n_samples=5)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_data, batch_sampler=train_batch_sampler, **kwargs)

print(2)

train_all()

model=nn2()
summary(model,(3,224,224))

