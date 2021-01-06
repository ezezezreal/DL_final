import torch
#torch.cuda.current_device()
import torch.nn as nn
from torchvision import datasets ,transforms
import torchvision
from matplotlib import pyplot as plt
import numpy as np
#from google.colab.patches import cv2_imshow
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tensorflow import summary
from torchvision.utils import make_grid
import torch.nn.functional as F
import os
import gc

#drive.mount('/content/drive')
trainRoot = r"Trail_dataset/Trail_dataset/train_data/"  
testRoot = r"Trail_dataset/Trail_dataset/test_data/"  
#trainRoot = "drive/MyDrive/Colab Notebooks/Trail_dataset/Trail_dataset/train_data/"  
#testRoot = "drive/MyDrive/Colab Notebooks/Trail_dataset/Trail_dataset/test_data/" 
#Load data set
train_data = datasets.ImageFolder(
    trainRoot,
    transform = transforms.Compose([transforms.ToTensor()])                         
)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=50,shuffle= True)



#show labels
# print(train_data.classes)
# print(train_data.class_to_idx)

#check availability of gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32*117*157, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 18)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 32*117*157)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN_Model()

#model.load_state_dict(torch.load('dlFinal_model.pth'))

LR = 0.0001
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
a = 0
epoches = 20
for epoch in range(epoches):  
    running_loss = 0.0
    a += 1
    print(a)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:    
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

PATH = 'dlFinal_model22.pth'
torch.save(model.state_dict(),'dlFinal_model33.pth',_use_new_zipfile_serialization=False)

#torch.save(model.state_dict(), PATH)
correct = 0
total = 0

del train_data
del train_loader
gc.collect()

test_data = datasets.ImageFolder(
    testRoot,
    transform = transforms.Compose([transforms.ToTensor()])                         
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=True)
with torch.no_grad():
    for data in test_data:
        images, labels = data
        images = torch.unsqueeze(images,0)
        outputs = model(images)
        top1 = outputs.argmax()
        if(top1 == labels):
            correct += 1
        total += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
