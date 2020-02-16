import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataset import Subset
import torchvision.datasets as dsets
import os
import sys
import matplotlib.pyplot as plt
import random

class ShakeShake(torch.autograd.Function):
  @staticmethod
  def forward(ctx,i1,i2):
    alpha = random.random()
    result = i1 * alpha + i2 * (1-alpha)
    return result
  @staticmethod
  def backward(ctx,grad_output):
    beta = random.random()
    return grad_output * beta, grad_output * (1-beta)

class ResidualPlainBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, padding=0):
        super(ResidualPlainBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size=3,  stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,  kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1_2 = nn.Conv2d(in_channels,  out_channels, kernel_size=3,  stride=stride, padding=1)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.conv2_2 = nn.Conv2d(out_channels, out_channels,  kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(out_channels)

        self.identity = nn.Identity()

        if in_channels != out_channels:
          self.down_avg1 = nn.AvgPool2d(kernel_size=1, stride=1)
          self.down_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=stride, padding=0)
          self.down_pad1 = nn.ZeroPad2d((1,0,1,0))
          self.down_avg2 = nn.AvgPool2d(kernel_size=1, stride=1)
          self.down_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=stride, padding=0)

    #down sampling時の処理が特殊 
    def shortcut(self,x):
      x = F.relu(x)
      h1 = self.down_avg1(x)
      h1 = self.down_conv1(h1)
      h2 = self.down_pad1(x[:,:,1:,1:])
      h2 = self.down_avg1(h2)
      h2 = self.down_conv2(h2)
      return torch.cat((h1,h2),axis=1)


    def forward(self, x):
      if self.training:
        #1つ目のResdual Block
          out = self.bn1(self.conv1(F.relu(x)))
          out = self.bn2(self.conv2(F.relu(out)))
          
        #2つ目のResidual Block
          out2 = self.bn1_2(self.conv1_2(F.relu(x)))
          out2 = self.bn2_2(self.conv2_2(F.relu(out2)))

          if self.in_channels != self.out_channels:
            output = self.shortcut(x) + ShakeShake.apply(out,out2)
          else:
            output = self.identity(x) + ShakeShake.apply(out,out2)
          
          return output
      else:
          out = self.bn1(self.conv1(F.relu(x)))
          out = self.bn2(self.conv2(F.relu(out)))
          
          out2 = self.bn1_2(self.conv1_2(F.relu(x)))
          out2 = self.bn2_2(self.conv2_2(F.relu(out2)))

          if self.in_channels != self.out_channels:
            output = self.shortcut(x) + (out+out2)*0.5
          else:
            output = self.identity(x) + (out+out2)*0.5
          
          return output

#[3,4,6,3]
class MyNet(nn.Module):
  def __init__(self):
      super(MyNet,self).__init__()

      self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

      self.layer1 = nn.Sequential(
        ResidualPlainBlock(16,32,1),
        ResidualPlainBlock(32,32,1),
        ResidualPlainBlock(32,32,1),
        ResidualPlainBlock(32,32,1)
      )
      self.layer2 = nn.Sequential(
        ResidualPlainBlock(32,64,2),
        ResidualPlainBlock(64,64,1),
        ResidualPlainBlock(64,64,1),
        ResidualPlainBlock(64,64,1)
        )
      self.layer3 = nn.Sequential(
        ResidualPlainBlock(64,128,2),
        ResidualPlainBlock(128,128,1),
        ResidualPlainBlock(128,128,1),
        ResidualPlainBlock(128,128,1),
        )
      self.avgpool = nn.MaxPool2d(kernel_size=8, stride=2, padding=1)
      self.fc = nn.Linear(128*4, 10)

  def forward(self,x):
    x = self.conv1(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.avgpool(x)
    x = x.view(-1,128*4)
    x = F.dropout(x,training=self.training)
    x = self.fc(x)
    return x


#画像の読み込み
batch_size = 128
train_data = dsets.CIFAR10(root='./tmp/cifar-10', train=True, download=False, transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) ]))
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_data = dsets.CIFAR10(root='./tmp/cifar-10', train=False, download=False, transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) ]))
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

net = MyNet().to(device)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.02
optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001)

loss = 0
correct = 0
train_list = []
test_list = []
max_acc = 0

#訓練・推論
for i in range(200):
  
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

    print('%d: %.3f'%(j+1,loss))

    _,predicted = torch.max(outputs.data,1)

    correct += (predicted == labels).sum()

  accuracy = 100.*correct / 50000
  print('\nepoch:%d Train Accuracy(%d/50000):%.2f'%(i+1,correct,accuracy))
  train_list.append(accuracy)

  correct = 0
  accuracy = 0.0
  net.eval()
 
  for j,data in enumerate(test_loader,0):

    inputs,labels = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = net(inputs)

    _,predicted = torch.max(outputs.data,1)

    correct += (predicted == labels).sum()

  accuracy = 100.*correct / 10000
  test_list.append(accuracy)

  print('epoch:%d Test Accuracy(%d/10000):%.2f\n'%(i+1,correct,accuracy))
  torch.save(net.state_dict(),'Weight'+str(i+1))

  scheduler.step()
  correct = 0

for i in range(len(test_list)):
  print('epoch:%d Accuracy:%.3f'%(i+1,test_list[i]))
print('max accuracy = %.3f'%(max(test_list)))

plt.plot(train_list,marker='x', label='train_acc')
plt.plot(test_list,marker='o', label='test_acc')
plt.show()
