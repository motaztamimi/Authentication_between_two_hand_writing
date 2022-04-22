import torch
import torch.nn as nn
import torch.nn.functional as F
from data_set import LinesDataSetTriplet, LinesDataSet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
class TriplteLoss(nn.Module):
    "Triplet loss function"
    def __init__(self, margin=3):
        super(TriplteLoss, self).__init__()
        self.margin = margin
        self.loss_function = nn.MarginRankingLoss(margin=2)
    def forward(self, anc, pos, neg):
        dist_a_p = F.pairwise_distance(anc, pos, 2)
        dist_a_n = F.pairwise_distance(anc, neg, 2)
        target = torch.FloatTensor(dist_a_p.size()).fill_(1)
        target = target.cuda()
        return loss_function(dist_a_n, dist_a_p, target)
        


def conv3x3(in_channels, out_channels, stride=1, padding=1, kernel_size=3):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=padding, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(1, 16, kernel_size=3, 
                     stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(16,eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.Max_pol = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        # self.layer4 = self.make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(64, 32))
        # self.fc1 = nn.Sequential(nn.Linear(128, num_classes), nn.Sigmoid())

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride, padding=0, kernel_size=1),
                nn.BatchNorm2d(out_channels,eps=1e-05))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)


    def forward_once(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.Max_pol(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = self.fc1(out)
        return out

    def forward(self, input1, input2, input3=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        if input3 != None:
            output3 = self.forward_once(input3)
            output = torch.stack((output1, output2, output3))
        else:
            output = torch.stack((output1, output2))
        return output



def triplet_train( model, loss_function, optimizer,train_loader,loss_history, epoch):
    model.train()
    batches_loss = []
    for _ , data in enumerate(train_loader):
        image1, image2, image3 = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image3 = image3.float().cuda()
        image1 = image1[:,None,:,:]
        image2 = image2[:,None,:,:]
        image3 = image3[:,None,:,:]

        output = model(image1, image2, image3)
        optimizer.zero_grad()
        loss = loss_function(output[0], output[1], output[2])
        loss_history.append(loss.cpu().item())
        batches_loss.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
        loss_history_for_epoch.append( sum(batches_loss) / len(batches_loss))


def test( model,test_loader):
    model.eval()
    for _ , data in enumerate(test_loader):
        image1, image2, image3 = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image3 = image3.float().cuda()
        image1 = image1[:, None, :, :]
        image2 = image2[:, None, :, :]
        image3 = image3[:, None, :, :]
        label = label.cuda()
        with torch.no_grad():
            output = model(image1, image2, image3)
            dist_a_p = F.pairwise_distance(output[0], output[1], 2)
            dist_a_n = F.pairwise_distance(output[0], output[2], 2)
            pred = (dist_a_n - dist_a_p).cpu().data
            acc_for_batches.append((pred > 0).sum()*1.0/dist_a_p.size()[0])
        

if __name__ == '__main__':
    train_line_data_set = LinesDataSetTriplet(csv_file="train_triplet.csv", root_dir="data_for_each_person", transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]))
    test_line_data_set = LinesDataSet(csv_file="Test_labels_for_arabic.csv", root_dir='data_for_each_person',  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]))
    train_line_data_loader = DataLoader(train_line_data_set, shuffle=True, batch_size=20)
    test_line_data_loader = DataLoader(test_line_data_set, shuffle=True, batch_size=1)
    my_model = ResNet(ResidualBlock, [2, 2, 2]).cuda()
    loss_function = TriplteLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
    
    
    for i in range(20):
        dist_with_label = []
        loss_history_for_epoch = []
        acc_for_batches = []
        print(f"train epoch: {i}")
        triplet_train(my_model, loss_function, optimizer, train_line_data_loader, loss_history=[], epoch=0)
        print(f'epoch loss: {sum(loss_history_for_epoch) / len(loss_history_for_epoch)}')
        print(f"test epoch: {i}")
        print(f"test acc: {(sum(acc_for_batches) / len(acc_for_batches))}")
        test(my_model, test_line_data_loader)
        dict_ = {i: 0 for i in list(np.arange(0.5, 20.5, 0.5))}
        for i in list(np.arange(0.5, 20.5, 0.5)):
            for dist, label in dist_with_label:
                if dist.item() <= i and label.item() == 0.:
                    dict_[i] += 1
                elif dist.item() > i and label.item() == 1.:
                    dict_[i] += 1 
        print(f"test results: {dict_}")            


    

        
        
