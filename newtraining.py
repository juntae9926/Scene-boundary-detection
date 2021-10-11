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
    image_dir = "/workspace/jt/places/places_16_211008"
    image_datasets = {x: datasets.ImageFolder(os.path.join(image_dir, x), data_transform[x])
                      for x in ['train', 'val']}
    dataloader = DataLoader(image_datasets['train', 'val'], batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print(dataset_sizes['train'], dataset_sizes['val'])

    # Model
    model = models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    params_to_update = model.parameters()



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.9)

    # Model visualization
    print(model)

    best_prec1 = 0
    for epoch in range(start_epoch, start_epoch + 200):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, start_epoch + 200))
        train(dataloader, model, criterion, optimizer, epoch)
        val(dataloader, model, criterion, optimizer, epoch, BATCH_SIZE)
        scheduler.step()




def train_model(model, dataloaders, criterion, optimizer, num_epochs=100, is_inspection=False):

