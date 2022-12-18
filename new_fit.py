import torch
import numpy as np


def fit(train_loader, val_loader, model,  optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
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
        train_loss, metrics = train_epoch(train_loader, model,  optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        accuracy,fa_rate, metrics = test_epoch(val_loader, model, cuda, metrics)
        accuracy /= len(val_loader)
        accuracy *=100
        fa_rate /= len(val_loader)
        fa_rate *=100

        message += '\nEpoch: {}/{}. Validation set: accuracy: {:.4f} fa_rate={:.4f}'.format(epoch + 1, n_epochs,
                                                                                            accuracy, fa_rate)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)


def train_epoch(train_loader, model,  optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

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

        for metric in metrics:
            metric(embeddings, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model,  cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            embeddings = model(*data)

            loss_outputs = batch_hard_triplet_loss(target,embeddings,0.2)

            pairwise_dist =_pairwise_distances(embeddings)
            labels_equal = target.unsqueeze(0) == target.unsqueeze(1)
            labels_unequal = target.unsqueeze(0) != target.unsqueeze(1)
            labels_equal = labels_equal.float()
            labels_unequal=labels_unequal.float()

            p_same=pairwise_dist*labels_equal
            p_diff=pairwise_dist*labels_unequal
            accept=torch.where(pairwise_dist<0.6,1,0)
            true_accept=torch.numel(torch.nonzero(p_same*accept))
            false_accept=torch.numel(torch.nonzero(p_diff*accept))
            accuracy=true_accept/torch.numel(torch.nonzero(p_same))
            fa_rate=false_accept/torch.numel(torch.nonzero(p_same))
            for metric in metrics:
                metric(embeddings, target, loss_outputs)

    return accuracy,fa_rate,metric
