import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.init as init
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset,DataLoader
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

#画像の読み込み
batch_size = 100
train_data = dsets.CIFAR10(root='./tmp/cifar-10', train=True, download=False, transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)]))
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_data = dsets.CIFAR10(root='./tmp/cifar-10', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_ft = models.resnet50(pretrained=True)
model_ft.fc = nn.Linear(model_ft.fc.in_features, 10)
net = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=0.00005)

loss, epoch_loss, count = 0, 0, 0
acc_list = []
loss_list = []

#訓練・推論
for i in range(50):

  #ここから学習
  net.train()

  for j,data in enumerate(train_loader,0):
    optimizer.zero_grad()

    #1:訓練データを読み込む
    inputs,labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    #2:計算する
    outputs = net(inputs)

    #3:誤差を求める
    loss = criterion(outputs,labels)

    #4:誤差から学習する
    loss.backward()
    optimizer.step()

    epoch_loss += loss
    count += 1
    print('%d: %.3f'%(j+1,loss))

  print('%depoch:mean_loss=%.3f\n'%(i+1,epoch_loss/count))
  loss_list.append(epoch_loss/count)

  epoch_loss, count = 0, 0
  correct,total = 0, 0
  accuracy = 0.0

  #ここから推論
  net.eval()

  for j,data in enumerate(test_loader,0):

    #テストデータを用意する
    inputs,labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    #計算する
    outputs = net(inputs)

    #予測値を求める
    _,predicted = torch.max(outputs.data,1)

    #精度を計算する
    correct += (predicted == labels).sum()
    total += batch_size

  accuracy = 100.*correct / total
  acc_list.append(accuracy)

  print('epoch:%d Accuracy(%d/%d):%f'%(i+1,correct,total,accuracy))
  torch.save(net.state_dict(),'Weight'+str(i+1))

plt.plot(acc_list)
plt.show(acc_list)
plt.plot(loss_list)
plt.show(loss_list)
