import os
import random
import argparse
import time
from datetime import datetime
from pytz import timezone
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from PIL import Image
import cv2
from siamese import Siamesenetwork, BCEloss

def threashold_sigmoid(t):
    """prob > 0.5 --> 1 else 0"""
    threashold = t.clone()
    threashold.data.fill_(0.5)
    return (t > threashold).float()

class LFWDataset(Dataset):

    def __init__(self, root_dir, path_file_dir, transform=None):
        self.root_dir = root_dir
        path_file = open(path_file_dir, 'r')
        data = []
        for line in path_file:
            line = line.strip()
            img1, img2, label = line.split(' ')
            label = int(label)
            data.append((img1, img2, label))
        self.data = data
        self.transform = transform
        path_file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, img2, label = self.data[idx]
        img1_file = Image.open(os.path.join(self.root_dir, img1))
        img2_file = Image.open(os.path.join(self.root_dir, img2))
        

        if self.transform:
            img1_file = self.transform(img1_file)
            img2_file = self.transform(img2_file)
        return (img1_file, img2_file, label)

    

def train(args):
    default_transform = transforms.Compose([
        transforms.Scale(128),
        transforms.ToTensor(),
    ])
    train_dataset = LFWDataset('./lfw', './train.txt', default_transform, args.randaug)
    print("Loaded {} training data.".format(len(train_dataset)))

    # # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               shuffle=True)

    siamese_net = Siamesenetwork()
    if args.cuda:
        siamese_net = siamese_net.cuda()

    # # Loss and Optimizer
    criterion = BCEloss()

    optimizer = torch.optim.SGD(siamese_net.parameters(), lr=0.1, momentum=0.9)

    # Train the Model
    num_epochs = args.epoch
    for epoch in range(num_epochs):
        for i, (img1_set, img2_set, labels) in enumerate(train_loader):

            if args.cuda:
                img1_set = img1_set.cuda()
                img2_set = img2_set.cuda()
                labels = labels.cuda()

            img1_set = Variable(img1_set)
            img2_set = Variable(img2_set)
            labels = Variable(labels.view(-1, 1).float())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            if args.contra_loss:
                output1, output2 = siamese_net(img1_set, img2_set)
                loss = criterion(output1, output2, labels)
                loss.backward()
                optimizer.step()
            else:
                output_labels_prob = siamese_net(img1_set, img2_set)
                loss = criterion(output_labels_prob, labels)
                loss.backward()
                optimizer.step()
        print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch+1, num_epochs, i+1, len(train_dataset)//128, loss.data[0]))

    # Training accuracy
    test_against_data(args, 'training', train_loader, siamese_net)

    return siamese_net

def test_against_data(args, label, dataset, siamese_net):
# Training accuracy
    siamese_net.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.0
    total = 0.0
    for img1_set, img2_set, labels in dataset:
        labels = labels.view(-1, 1).float()
        if args.cuda:
            img1_set = img1_set.cuda()
            img2_set = img2_set.cuda()
            labels = labels.cuda()
        img1_set = Variable(img1_set)
        img2_set = Variable(img2_set)
        labels = Variable(labels)

        
        output_labels_prob = siamese_net(img1_set, img2_set)
        output_labels = threashold_sigmoid(output_labels_prob)

        if args.cuda:   
            output_labels = output_labels.cuda()
        total += labels.size(0)
        correct += (output_labels == labels).sum().data[0]

    print('Accuracy of the model on the {} {} images: {} %%'.format(total, label, (100 * correct / total)))

def test(args, siamese_net=None):
    if not siamese_net:
        saved_model = torch.load(args.model_file)
        siamese_net = Siamesenetwork(args.contra_loss)
        siamese_net.load_state_dict(saved_model)

    if args.cuda:
        siamese_net = siamese_net.cuda()

    default_transform = transforms.Compose([
        transforms.Scale(128),
        transforms.ToTensor(),
    ])
    test_dataset = LFWDataset('./lfw', './test.txt', default_transform)
    print("Loaded {} test data.".format(len(test_dataset)))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False)

    test_against_data(args, "testing", test_loader, siamese_net)