import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from skimage import io
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import os
import numpy as np
from data_set import LinesDataSet

class ContrastiveLoss(nn.Module):
    "Contrastive loss function"
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
            
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance((output1, output2),keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive



def train ( model , loss_function, optimizer,train_loader,device):

    model.train()
    for indx , data in enumerate(train_loader):
        image1, image2, label = data
        image1 = image1
        image2 = image2
        image1 = image1[:,None,:,:]
        image2 = image2[:,None,:,:]
        label = label   
        optimizer.zero_grad()
        output1 = model(image1)
        output2 = model(image2)
        loss = loss_function(output1, output2, label)
        print(loss.item())
        loss.backward() 
        optimizer.step()
        


if __name__ == "__main__":
    line_data_set = LinesDataSet(csv_file="Train_Labels.csv",root_dir="data_for_each_person",transform=transforms.Compose([transforms.ToTensor()]))
    line_data_loader = DataLoader(line_data_set,shuffle=True,batch_size=100)
    loss_function = ContrastiveLoss()
    model = torchvision.models.resnet18(pretrained = False)
    model.conv1=torch.nn.Conv2d(1,64,7,2,3,bias=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, loss_function, optimizer, line_data_loader, "cuda")
  


