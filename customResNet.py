import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from data_set import LinesDataSet
from torch.utils.data import DataLoader
from way_2_model import Net

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

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
        self.in_channels = 64
        self.conv = conv3x3(1, 64)
        self.bn = nn.BatchNorm2d(64,eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.Max_pol = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512, 200), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(800, num_classes), nn.Sigmoid())

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
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
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        h_ = output1 * output2
        dist_ = torch.pow((output1 - output2), 2)
        V_ = torch.cat((output1, output2, dist_, h_), dim=1)
        output = self.fc1(V_)
        return output


if __name__ == '__main__':
    # model  = torchvision.models.resnet18(pretrained = False)
    # print(model)
    model = ResNet(ResidualBlock, [2, 2, 2, 2])           
    # my_model = Net()
    print(model)

# model = ResNet(ResidualBlock, [2, 2, 2]).to('cuda')
# train_line_data_set = LinesDataSet(csv_file="Train_Labels.csv", root_dir="data_for_each_person", transform=transforms.Compose([transforms.ToTensor()]))
# train_line_data_loader = DataLoader(train_line_data_set,shuffle=True,batch_size=14)

# for _ , data in enumerate(train_line_data_loader):
#         image1, image2, label = data
#         image1 = image1.float().cuda()
#         image2 = image2.float().cuda()
#         image1 = image1[:,None,:,:]
#         image2 = image2[:,None,:,:]
#         label = label.cuda()
#         out = model(image1, image2)

# a = torchvision.models.resnet18(pretrained = False)

# print(model)




# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# from data_set import LinesDataSet
# from torch.utils.data import DataLoader


# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
#                      stride=stride, padding=1, bias=False)

# # Residual block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out

# # ResNet
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_channels = 16
#         self.conv = conv3x3(1, 16)
#         self.bn = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self.make_layer(block, 16, layers[0])
#         self.layer2 = self.make_layer(block, 32, layers[1], 2)
#         self.layer3 = self.make_layer(block, 64, layers[2], 2)
#         self.avg_pool = nn.AvgPool2d(8)
#         self.fc = nn.Linear(64, num_classes)

#     def make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if (stride != 1) or (self.in_channels != out_channels):
#             downsample = nn.Sequential(
#                 conv3x3(self.in_channels, out_channels, stride=stride),
#                 nn.BatchNorm2d(out_channels))
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels
#         for i in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         print(x.size())
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         print(out.size())
#         out = self.layer1(out)
#         print(out.size())
#         out = self.layer2(out)
#         print(out.size())
#         out = self.layer3(out)
#         print(out.size())
#         out = self.avg_pool(out)
#         print(out.size())
#         out = out.view(out.size(0), -1)
#         print(out.size())
#         out = self.fc(out)
#         return out


# model = ResNet(ResidualBlock, [2, 2, 2]).to('cuda')
# train_line_data_set = LinesDataSet(csv_file="Train_Labels.csv", root_dir="data_for_each_person", transform=transforms.Compose([transforms.ToTensor()]))
# train_line_data_loader = DataLoader(train_line_data_set,shuffle=True,batch_size=5)

# for _ , data in enumerate(train_line_data_loader):
#         image1, image2, label = data
#         image1 = image1.float().cuda()
#         image2 = image2.float().cuda()
#         image1 = image1[:,None,:,:]
#         image2 = image2[:,None,:,:]
#         label = label.cuda()
#         out = model(image1)

# a = torchvision.models.resnet18(pretrained = False)

# print(a)
