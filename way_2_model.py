from torch import nn
import torch
import torchvision


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = torchvision.models.resnet18(pretrained = False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, 3, bias=False)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(num_features, 200), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(800, 2), nn.Sigmoid())

    def forward_once(self, x):
        output = self.cnn(x)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        h_ = output1 * output2
        dist_ = torch.pow((output1 - output2), 2)
        V_ = torch.cat((output1, output2, dist_, h_), dim=1)
        output = self.fc1(V_)
        return output

 