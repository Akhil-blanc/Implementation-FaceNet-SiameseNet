from model import *
from data_generator import *

import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

def train(train_loader, num_epochs, device):
    # default_transform = transforms.Compose([
    #     transforms.Scale(128),
    #     transforms.ToTensor(),
    # ])
     
    # # Data Loader (Input Pipeline)
    criterion = nn.BCELoss()
    siamese_net = SiameseNetwork()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = criterion.to(device)
    siamese_net = siamese_net.to(device)
   
    # # Loss and Optimizer

    optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)
    # lambda1 = lambda epoch: 0.99
    # # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # # Train the Model
   
    for epoch in range(num_epochs):
        counter = 0
        for i, (img1_set, img2_set, labels) in enumerate(train_loader):

            img1_set = Variable(img1_set)
            img2_set = Variable(img2_set)
            labels = Variable(labels.view(-1, 1).float())
            img1_set = img1_set.to(device)
            img2_set = img2_set.to(device)
            labels = labels.to(device)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            # if args.contra_loss:
            #     output1, output2 = siamese_net(img1_set, img2_set)
            #     loss = criterion(output1, output2, labels)
            #     loss.backward()
            #     optimizer.step()
            # else:
            output_labels_prob = siamese_net(img1_set, img2_set)
            loss = criterion(output_labels_prob, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            print('Epoch [%d/%d] Iter %d Loss: %.4f' % (epoch+1, num_epochs, i+1, loss.item()))
        # scheduler.step()
    # Training accuracy
    # test_against_data(args, 'training', train_loader, siamese_net)

    return siamese_net

train(train_loader, 1, 'cuda')