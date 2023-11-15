import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.models import resnet101
from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm

name = 'resnet101_pretrained_catdog'
# model = resnet101(weights=None)
model = resnet101(weights='ResNet101_Weights.DEFAULT')
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

class CatDogDataset(Dataset):
    def __init__(self, img_path, label, transforms = None):
        super().__init__()
        self.img_path = img_path
        self.label = torch.tensor(label, dtype = torch.float32)
        self.transforms = transforms
        
    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        img = img.resize((224, 224))
        img = self.transforms(img)
        label = self.label[idx]
        return img, label
        
    def __len__(self):
        return len(self.label)
    
def get_label(dir_list):
    path = []
    label = []
    for i in dir_list:
        for root, dir, files in os.walk(i):
            for file in files:
                if file.endswith("jpg"):
                    path.append(i + file)
                    if 'cat' in file:
                        label.append(0)
                    elif 'dog' in file:
                        label.append(1)
    return path, label

train_path = ['train/cat/', 'train/dog/']
train_image_path, train_label = get_label(train_path)
train_dataset = CatDogDataset(img_path = train_image_path, label = train_label, transforms = get_train_transform())

valid_path = ['valid/cat/', 'valid/dog/']
valid_image_path, valid_label = get_label(train_path)
valid_dataset = CatDogDataset(img_path = valid_image_path, label = valid_label, transforms = get_train_transform())

train_data_loader = DataLoader(
    dataset = train_dataset,
    num_workers= 4,
    batch_size = 16,
    shuffle = True
)

val_data_loader = DataLoader(
    dataset = valid_dataset,
    num_workers= 4,
    batch_size = 16,
    shuffle = True
)

def accuracy(preds, trues):
    ### Converting preds to 0 or 1
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    ### Calculating accuracy by comparing predictions with true labels
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    ### Summing over all correct predictions
    acc = np.sum(acc) / len(preds)
    return (acc * 100)
    
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
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
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
        labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
        
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

#Linear probing
# for param in model.parameters():
#     param.requires_grad = False

# Modifying Head - classifier
model.fc = nn.Sequential(
    nn.Linear(2048, 1, bias = True),
    nn.Sigmoid()
)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

# Learning Rate Scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)

#Loss Function
criterion = nn.BCELoss()

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
    plt.savefig(f'result/catdog/acc/acc_{name}.png')
    plt.clf()
    plt.plot(train_loss, label = 'train_loss')
    plt.plot(valid_loss, label = 'valid_loss')
    plt.legend()
    plt.title(f'Loss_{name}')
    plt.savefig(f'result/catdog/loss/loss_{name}.png')
    plt.close()

    data = {'train_acc': train_acc, 'train_loss': train_loss, 'valid_acc': valid_acc, 'valid_loss': valid_loss}
    df = pd.DataFrame(data)
    df.to_csv(f'result/catdog/{name}.csv', index=False)