from models import *
from helpers import *
from loss import *
from train_helpers import *

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch import optim

from torchsummary import summary
from torch.optim import lr_scheduler

#checking if cuda/gpu is available to use
cuda = torch.cuda.is_available()

data = datasets.CelebA(root = "./data", download = True)

# eps = 1e-8 

#pixel size dictionary for models
pix_size = {'nn1':(220, 220), 'nn2':(224, 224), 'nn3':(160, 160), 'nn4':(96,96), 'nns1':(165, 165)}

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
        print("Creating model :", i)
        print()
        
        model = create_model(i)
        
        if cuda:
            model.cuda()
        
        print("Creating train data...")
        print()

        train_data = torchvision.datasets.CelebA(root = "./data", split = "train",target_type='identity', download = True, transform = transforms.Compose([transforms.Resize((pix_size[i][0],pix_size[i][1])), transforms.PILToTensor(), transforms.ConvertImageDtype(dtype = torch.float32)]))
        
        print("Created train data...")
        print()

        print("Creating test data...")
        print()


        test_data = torchvision.datasets.CelebA(root = "./data", split = "valid",target_type='identity', download = True, transform = transforms.Compose([transforms.Resize((pix_size[i][0],pix_size[i][1])), transforms.PILToTensor(), transforms.ConvertImageDtype(dtype = torch.float32)]))

        print("Created test data...")
        print()
        
        print("Creating samples...")
        print()
        
        train_batch_sampler = BalancedBatchSampler(train_data, n_classes=5, n_samples=5)
        test_batch_sampler = BalancedBatchSampler(test_data, n_classes=5, n_samples=5)

        print("Created samples...")
        print()
        
        print("Loading Data...")
        print()
        
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        train_loader = DataLoader(train_data, batch_sampler=train_batch_sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_sampler=test_batch_sampler, **kwargs)

        print("Data Loaded...")
        print()

        optimizer = optim.Adagrad(model.parameters(), lr = 0.003 )
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
        n_epochs = 1
        log_interval = 5

        fit(train_loader, test_loader, model, optimizer, scheduler, n_epochs, cuda, log_interval)
                
        summary(model,(3,pix_size[i][0],pix_size[i][1]))
                
        torch.save(model.state_dict(), "FaceNet/saved_models/" + i + ".pkl")
        
        input_yn = input("Continue Signal")

train_all()



