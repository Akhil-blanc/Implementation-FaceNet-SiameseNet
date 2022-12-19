from loss import *
from helpers import *

import torch
import numpy as np


def fit(train_loader, val_loader, model,  optimizer, scheduler, n_epochs, cuda, log_interval,
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss = train_epoch(train_loader, model,  optimizer, cuda, log_interval)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

        accuracy,fa_rate= test_epoch(val_loader, model, cuda)
        accuracy /= len(val_loader)
        accuracy *=100
        fa_rate /= len(val_loader)
        fa_rate *=100

        message += '\nEpoch: {}/{}. Validation set: accuracy: {:.4f} fa_rate={:.4f}'.format(epoch + 1, n_epochs,
                                                                                            accuracy, fa_rate)

        print(message)


def train_epoch(train_loader, model,  optimizer, cuda, log_interval):
    """"
    Helper function for training loop
    """
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        embeddings = model(*data)


        loss_outputs = batch_hard_triplet_loss(target,embeddings,0.2)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss


def test_epoch(val_loader, model,  cuda):
    """"
    Helper function for testing loop
    """
    with torch.no_grad():
        model.eval()
        accuracy = 0
        fa_rate = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            embeddings = model(*data)

            pairwise_dist =pairwise_distances(embeddings)
            labels_equal = target.unsqueeze(0) == target.unsqueeze(1)
            labels_unequal = target.unsqueeze(0) != target.unsqueeze(1)
            labels_equal = labels_equal.float()
            labels_unequal=labels_unequal.float()

            p_same=pairwise_dist*labels_equal
            p_diff=pairwise_dist*labels_unequal
            accept=torch.where(pairwise_dist<0.6,1,0)
            true_accept=torch.numel(torch.nonzero(p_same*accept))
            false_accept=torch.numel(torch.nonzero(p_diff*accept))
            accuracy+=true_accept/torch.numel(torch.nonzero(p_same))
            fa_rate+=false_accept/torch.numel(torch.nonzero(p_same))

    return accuracy,fa_rate
