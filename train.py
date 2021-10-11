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
    trainloader = DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    valloader = DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print(dataset_sizes['train'], dataset_sizes['val'])

    # Tensorboard
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('Image_data', img_grid)

    # Training data visualization
    for img, data in trainloader:
        print("img shape", img.shape)  # torch.Size([8, 3, 224, 224]) 형태로 나와서 (224, 224, 3) tensor로 바꿔줘야 출력가능
        imshow(img)
        break

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
        train(trainloader, model, criterion, optimizer, epoch)
        val(valloader, model, criterion, optimizer, epoch, BATCH_SIZE)
        scheduler.step()


# Training
def train(trainloader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    top1.reset()
    top5.reset()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
    print("Train Loss: {}".format(train_loss/1000))
    print("Train top1 accuracy: {}".format(top1.avg))
    print("Train top5 accuracy: {}".format(top5.avg))

# Validation
def val(valloader, model, criterion, optimizer, epoch, BATCH_SIZE):
    global best_prec1
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    top1.reset()
    top5.reset()
    class_correct = list(0. for _ in range(15))
    class_total = list(0. for _ in range(15))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            writer.add_scalar("Loss/test", loss, epoch)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            c = (predicted == targets)
            for j in range(len(targets.data)):
                target = targets[j]
                class_correct[target] += c[j].item()
                class_total[target] += 1

            test_grid = torchvision.utils.make_grid(inputs)
            writer.add_image("test", test_grid, epoch)

        for i in range(15):
            writer.add_scalar("accuracy of {}".format(class_names[i]), 100 * class_correct[i] / class_total[i], epoch)
            print("Accuracy of %5s : %2d %%" % (class_names[i], 100*class_correct[i]/class_total[i]))

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
    print("Val Loss: {}".format(val_loss/1000))

    # Save checkpoint.
    acc = 100.*correct/total
    print("TOP-1 ACCURACY = ", acc)
    if acc > best_prec1:
        print('Saving..')
        state = {
            'epoch' : epoch + 1,
            'arch' : 'resnet50',
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_prec1': best_prec1,
            'batch_size': BATCH_SIZE
        }
        torch.save(state, '/workspace/jt/model/paper-{}.pth.tar'.format(datetime.now().strftime('%Y%m%d-%H')))
        best_prec1 = acc
        print("Best val accuracy: {}".format(best_prec1))

    if epoch % 10 == 0:
        print('Saving..')
        state = {
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1,
            'batch_size': BATCH_SIZE
        }
        torch.save(state, '/workspace/jt/model/paper-{}.pth.tar'.format(datetime.now().strftime('%Y%m%d-%H')))
        best_prec1 = acc
        print("Best val accuracy: {}".format(best_prec1))


def imshow(img):
    img = torchvision.utils.make_grid(img.cpu().detach())
    img = (img+1)/2
    img = img.squeeze()
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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


if __name__ == '__main__':
    main()