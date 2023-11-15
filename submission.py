import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.models import resnet50, resnet101

from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm

model = resnet50(weights=None)
weight = "resnet50_pretrained_catdog.pth"
model.fc = nn.Sequential(
    nn.Linear(2048, 1, bias = True),
    nn.Sigmoid()
)
model.load_state_dict(torch.load(f'weight/{weight}'))
device = 'cuda'
model.to(device)
submission = pd.DataFrame(columns = ['id', 'label'])

def get_train_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])

for i in range(1,501):
    img_ori = Image.open(f'test/{i}.jpg')
    img = img_ori.resize((224, 224))
    img = get_train_transform()(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    pred = model(img)
    pred = pred.cpu()
    pred = pred.detach().numpy()
    pred = pred[0][0]
    submission.loc[i-1, 'id'] = i
    if pred > 0.5:
        print('dog')
        img_ori.save(f'output/dog_{i}.jpg')
        submission.loc[i-1, 'label'] = 1
    else:
        print('cat')
        img_ori.save(f'output/cat_{i}.jpg')
        submission.loc[i-1, 'label'] = 0

submission.to_csv('submission.csv', index = False)