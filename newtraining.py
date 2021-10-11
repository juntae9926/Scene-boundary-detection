import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import copy

import models

# Using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Summary
writer = SummaryWriter("/workspace/jt/model/{}_writer/{}".format('train', datetime.now().strftime('%Y%m%d-%H')))

# classes list
class_names = open('classes.txt', 'r').read().split('\n')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def main():
    global top1, top5, best_prec1



    # Parameter
    BATCH_SIZE = 32
    lr = 0.1

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Initial argument
    top1 = AverageMeter()
    top5 = AverageMeter()
    start_epoch = 0

    # Data Load
    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    image_dir = "/workspace/jt/places/places_16_210904"
    image_datasets = {x: datasets.ImageFolder(os.path.join(image_dir, x), data_transform[x]) for x in ['train', 'val']}
    dataloader = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=1) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("trainset: {}, testset: {}".format(dataset_sizes['train'], dataset_sizes['val']))

    # Model
    model, feature_extract = models.resnet50(pretrained=True, feature_extract=True)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.9)

    # Model visualization
    print(model)

    train(model, dataloader, criterion, optimizer, num_epochs=100)
    scheduler()




def train(model, dataloader, criterion, optimizer, num_epochs=100):
    since = time.time()

    val_acc_history = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, targets) in enumerate(dataloader[phase]):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    _, pred = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred == targets.data)

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_write = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_write)
    return model, val_acc_history

if __name__ == '__main__':
    main()
