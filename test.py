import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataset import Subset
import torchvision.datasets as dsets

import os
import sys
from skimage import io
from skimage import color
from skimage.transform import rescale

from PIL import Image

import matplotlib.pyplot as plt

import random

class ShakeShake(torch.autograd.Function):
  @staticmethod
  def forward(ctx, i1, i2):
    alpha = random.random()
    result = i1 * alpha + i2 * (1-alpha)

    return result
  @staticmethod
  def backward(ctx, grad_output):
    beta  = random.random()

    return grad_output * beta, grad_output * (1-beta)

class ResidualBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBottleneckBlock, self).__init__()

        bottleneck_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels,  bottleneck_channels, kernel_size=1,  stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,  kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0,  bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv1_2 = nn.Conv2d(in_channels,  bottleneck_channels, kernel_size=1,  stride=1, padding=0, bias=False)
        self.bn1_2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2_2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,  kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3_2 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0,  bias=False)
        self.bn3_2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()   # identity mapping
        if in_channels != out_channels:   # downsampling
            self.shortcut.add_module('conv',  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
      if self.training:
          out = F.relu(self.bn1(self.conv1(x)))
          out = F.relu(self.bn2(self.conv2(out)))
          out = self.bn3(self.conv3(out))
          
          out2 = F.relu(self.bn1_2(self.conv1_2(x)))
          out2 = F.relu(self.bn2_2(self.conv2_2(out2)))
          out2 = self.bn3_2(self.conv3_2(out2))

          output = self.shortcut(x) + ShakeShake.apply(out,out2)
          
          #output = out + self.shortcut(x)
          return F.relu(output)
      else:
          out = F.relu(self.bn1(self.conv1(x)), inplace=True)
          out = F.relu(self.bn2(self.conv2(out)), inplace=True)
          out = self.bn3(self.conv3(out))
          
          out2 = F.relu(self.bn1_2(self.conv1_2(x)), inplace=True)
          out2 = F.relu(self.bn2_2(self.conv2_2(out2)), inplace=True)
          out2 = self.bn3_2(self.conv3_2(out2))

          output = self.shortcut(x) + out*0.5 + out2*0.5
          
          #output = out + self.shortcut(x)
          return F.relu(output)
#[3,4,6,3]
class HoshiryuNet(nn.Module):
  def __init__(self):
      super(HoshiryuNet,self).__init__()

      self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
      self.bn1 = nn.BatchNorm2d(64)
      self.relu = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

      self.layer1 = nn.Sequential(
        ResidualBottleneckBlock(64,64,1),
        ResidualBottleneckBlock(64,64,1),
        ResidualBottleneckBlock(64,64,1)
      )
      self.layer2 = nn.Sequential(
        ResidualBottleneckBlock(64,128,2),
        ResidualBottleneckBlock(128,128,1),
        ResidualBottleneckBlock(128,128,1),
        ResidualBottleneckBlock(128,128,1)
        )
      self.layer3 = nn.Sequential(
        ResidualBottleneckBlock(128,256,2),
        ResidualBottleneckBlock(256,256,1),
        ResidualBottleneckBlock(256,256,1),
        ResidualBottleneckBlock(256,256,1),
        ResidualBottleneckBlock(256,256,1),
        ResidualBottleneckBlock(256,256,1)
        )
      self.layer4 = nn.Sequential(
        ResidualBottleneckBlock(256,512,2),
        ResidualBottleneckBlock(512,512,1),
        ResidualBottleneckBlock(512,512,1)
        )

      self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
      self.fc = nn.Linear(512 * 4, 10)

  def forward(self,x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(-1,512*4)
    x = self.fc(x)
    return x
  def weight_initializer(self):
    for m in self.modules():
        if isinstance(m,nn.Conv2d):
           init.kaiming_uniform_(m.weight.data,nonlinearity='relu')


#画像の読み込み
batch_size = 100
train_data = dsets.CIFAR10(root='./tmp/cifar-10', train=True, download=False, transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)]))
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_data = dsets.CIFAR10(root='./tmp/cifar-10', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

net = HoshiryuNet().to(device)
net.weight_initializer()

criterion = nn.CrossEntropyLoss()


learning_rate = 0.01
optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[650], gamma=0.1)

loss = 0
loss100 = 0
count = 0
acc_list = []
loss_list = []
max_acc = 0

#訓練・推論
for i in range(1000):
  
  net.train()
  
  for j,data in enumerate(train_loader,0):
    optimizer.zero_grad()
    inputs,labels = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs)

    loss = criterion(outputs,labels)

    loss.backward()
    optimizer.step()

    scheduler.step()

    loss100 += loss
    count += 1
    print('%d: %.3f'%(j+1,loss))

  print('%depoch:mean_loss=%.3f\n'%(i+1,loss100/count))
  loss_list.append(loss100/count)

  loss100 = 0
  count = 0
  correct = 0
  total = 0
  accuracy = 0.0
  net.eval()
 
  for j,data in enumerate(test_loader,0):

    inputs,labels = data

    param = torch.load('Weight'+str(i+1))
    net.load_state_dict(param)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs)

    _,predicted = torch.max(outputs.data,1)

    correct += (predicted == labels).sum()
    total += batch_size

  accuracy = 100.*correct / total
  acc_list.append(accuracy)

  print('epoch:%d Accuracy(%d/%d):%f'%(i+1,correct,total,accuracy))
  torch.save(net.state_dict(),'Weight'+str(909+i+1))

for i in range(len(acc_list)):
  print('epoch:%d Accuracy:%.3f'%(i+1,acc_list[i]))

plt.plot(acc_list)
plt.show(acc_list)
plt.plot(loss_list)
plt.show(loss_list)