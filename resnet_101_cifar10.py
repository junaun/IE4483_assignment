import os
import cv2
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.models import resnet101

from sklearn.model_selection import train_test_split

from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision.datasets import CIFAR10

def modify_resnet_cifar10(model):
    # Change the first convolutional layer to 3x3 kernel and stride 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove the max pooling layer
    model.maxpool = nn.Identity()

    # Change the number of output features in the final fully connected layer to 10
    model.fc = nn.Sequential(
        nn.Linear(2048, 10, bias=True),
    )
    return model

name = 'resnet101_notpretrained_cifar10'
# model = resnet101(weights = 'ResNet101_Weights.DEFAULT')
model = resnet101(weights = None)
model = modify_resnet_cifar10(model)
device = 'cuda'
def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])
    
def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])

# Download and load the training data
train_data = CIFAR10(root='./data', train=True, download=True, transform=get_train_transform())

# Download and load the validation data
val_data = CIFAR10(root='./data', train=False, download=True, transform=get_val_transform())
train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=64, shuffle=False)

def accuracy(preds, trues):
    # Get the predicted classes
    _, predicted = torch.max(preds, 1)
    
    # Calculate the number of correct predictions
    correct = (predicted == trues).sum().item()

    # Calculate the accuracy
    acc = correct / len(trues)

    return acc * 100
    
def train_one_epoch(train_data_loader):
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for images, labels in tqdm(train_data_loader):
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)

        #Reseting Gradients
        optimizer.zero_grad()
        
        #Forward
        preds = model(images)
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
        
        #Backward
        _loss.backward()
        optimizer.step()
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    return epoch_loss, epoch_acc, total_time
        
def val_one_epoch(val_data_loader, best_val_acc):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for images, labels in val_data_loader:
        
        #Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward
        preds = model(images)
        
        #Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Saving best model
    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(),f"weights/{name}.pth")
        
    return epoch_loss, epoch_acc, total_time, best_val_acc

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

# Learning Rate Scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)

#Loss Function
criterion = nn.CrossEntropyLoss()

# Loading model to device
model.to(device)

# No of epochs 
epochs = 15

best_val_acc = 0
if __name__ == '__main__':
    train_acc = []
    train_loss = []
    valid_acc = []
    valid_loss = []
    for epoch in tqdm(range(epochs)):
        
        ###Training
        loss, acc, _time = train_one_epoch(train_data_loader)
        train_acc.append(acc) 
        train_loss.append(loss)
        #Print Epoch Details
        print("\nTraining")
        print("Epoch {}".format(epoch+1))
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
        print("Time : {}".format(round(_time, 4)))
        
        ###Validation
        loss, acc, _time, best_val_acc = val_one_epoch(val_data_loader, best_val_acc)
        valid_acc.append(acc) 
        valid_loss.append(loss)
        #Print Epoch Details
        print("\nValidating")
        print("Epoch {}".format(epoch+1))
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
        print("Time : {}".format(round(_time, 4)))
        
    plt.plot(train_acc, label = 'train_acc')
    plt.plot(valid_acc, label = 'valid_acc')
    plt.legend()
    plt.title(f'Accuracy_{name}')
    plt.savefig(f'result/cifar10/acc/acc_{name}.png')
    plt.clf()
    plt.plot(train_loss, label = 'train_loss')
    plt.plot(valid_loss, label = 'valid_loss')
    plt.legend()
    plt.title(f'Loss_{name}')
    plt.savefig(f'result/cifar10/loss/loss_{name}.png')
    plt.close()

    data = {'train_acc': train_acc, 'train_loss': train_loss, 'valid_acc': valid_acc, 'valid_loss': valid_loss}
    df = pd.DataFrame(data)
    df.to_csv(f'result/cifar10/{name}.csv', index=False)
