import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import sys
import os
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataSets.data_set import LinesDataSetTriplet, LinesDataSetTripletWithLabel



class TripletLoss(nn.Module):
    "Triplet loss function"
    def __init__(self, margin=2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_function = nn.MarginRankingLoss(margin=self.margin)
    def forward(self, anc, pos, neg):
        dist_a_p = F.pairwise_distance(anc, pos, 2)
        dist_a_n = F.pairwise_distance(anc, neg, 2)
        target = torch.FloatTensor(dist_a_p.size()).fill_(1)
        target = target.cuda()
        return self.loss_function(dist_a_n, dist_a_p, target)
        

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
        self.in_channels = 32
        self.conv = nn.Conv2d(1, 32, kernel_size=3, 
                     stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(32,eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.Max_pol = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = self.make_layer(block, 32, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[1], 2)
        self.layer3 = self.make_layer(block, 128, layers[2], 2)
        # self.layer4 = self.make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(64, 32))

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
        out = self.fc1(out)
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


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.cnn = torchvision.models.resnet18(pretrained = False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, 3, bias=False)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_features, 200), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(200, 128))
    def forward_once(self, x):
        output = self.cnn(x)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        output = torch.stack((output1, output2, output3))
        return output

 
def triplet_train(model, loss_function, optimizer,train_loader):
    model.train()
    batches_loss = []
    for _ , data in enumerate(train_loader):
        image1, image2, image3, _ = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image3 = image3.float().cuda()
        image1 = image1[:,None,:,:]
        image2 = image2[:,None,:,:]
        image3 = image3[:,None,:,:]

        output = model(image1, image2, image3)
        optimizer.zero_grad()
        loss = loss_function(output[0], output[1], output[2])
        batches_loss.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
        loss_history_for_epoch.append( sum(batches_loss) / len(batches_loss))


def triplet_test(model,test_loader, train_flag, loss_function, y_pred=None, y_true=None):
    model.eval()
    batches_loss = []
    for _ , data in enumerate(test_loader):
        image1, image2, image3, label = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image3 = image3.float().cuda()
        label  = label.cuda()
        image1 = image1[:, None, :, :]
        image2 = image2[:, None, :, :]
        image3 = image3[:, None, :, :]
        with torch.no_grad():
            output = model(image1, image2, image3)
            loss = loss_function(output[0], output[1], output[2])
            batches_loss.append(loss.cpu().item())
            dist_a_p = F.pairwise_distance(output[0], output[1], 2)
            dist_a_n = F.pairwise_distance(output[0], output[2], 2)
            for indx, i in enumerate((0, 0.5, 1, 1.5, 2)):
                pred = (dist_a_n - dist_a_p - i).cpu().data
                if train_flag:
                    acc_for_batches_train[indx].append(((pred > 0).sum()*1.0/dist_a_p.size()[0]).item())
                    loss_history_for_epoch_test.append(sum(batches_loss) / len(batches_loss))
                else:
                    label_as_list = list(map(lambda x:int(x[0]), label.cpu().data.tolist()))
                    predeict_as_list = pred.tolist()
                    y_pred[indx].extend(np.where(np.array(predeict_as_list) > 0, 1, 0))
                    y_true[indx].extend(label_as_list)
                    acc_to_add = (sum((np.array(predeict_as_list) > 0) == np.array(label_as_list)) / dist_a_p.size()[0]) 
                    acc_for_batches[indx].append(acc_to_add)
        

if __name__ == '__main__':
    writer = SummaryWriter("../runs/custom_resnet_hebrew_25K_without_WD")
    train_line_data_set = LinesDataSetTripletWithLabel(csv_file="../train_labels_for_hebrew_triplet.csv", root_dir="../data2_for_each_person", transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]))
    test_line_data_set = LinesDataSetTripletWithLabel(csv_file="../test_labels_for_hebrew_triplet.csv", root_dir='../data2_for_each_person',  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]))
    train_line_data_loader = DataLoader(train_line_data_set, shuffle=True, batch_size=30)
    test_line_data_loader = DataLoader(test_line_data_set, shuffle=True, batch_size=30)
    train_line_data_loader_for_test = DataLoader(train_line_data_set,shuffle=True, batch_size=30)

    torch.manual_seed(17)
    my_model = ResNet(ResidualBlock, [2, 2, 2]).cuda()
    #my_model = ResNet18().cuda()
    loss_function = TripletLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
    # my_model.load_state_dict(torch.load(r"C:\Users\FinalProject\Desktop\backup_models\Triplet\custom_resnet_TripletLoss_25K_many_margins_hebrew_without_weight_decay\model_epoch_30.pt", map_location='cuda:0'))
    for i in range(30):
        print('epoch number: {}'.format(i + 1))
        loss_history_for_epoch = []
        loss_history_for_epoch_test = []
        acc_for_batches = [[] for _ in range(5)]
        acc_for_batches_train = [[] for _ in range(5)]
        y_pred = [[] for _ in range(5)]
        y_true = [[] for _ in range(5)]
        triplet_train(my_model, loss_function, optimizer, train_line_data_loader)
        torch.save(my_model.state_dict(), f'../model_epoch_{i + 1}.pt')
        print(f'epoch loss: {sum(loss_history_for_epoch) / len(loss_history_for_epoch)}')

        print('Testing on Train Data_set...')
        triplet_test(my_model, train_line_data_loader_for_test, train_flag=True, loss_function=loss_function, y_true=y_true, y_pred=y_pred)
        for index , i1 in enumerate((0, 0.5, 1, 1.5, 2)):
            print(f"train_acc_{i1}: ",100* (sum(acc_for_batches_train[index]) /  len(acc_for_batches_train[index])) )
        writer.add_scalars("train_acc",{str(value) : (sum(acc_for_batches_train[indx]) /  len(acc_for_batches_train[indx])) for indx, value in enumerate((0, 0.5, 1, 1.5, 2))},i)

        print('Testing on Test Data_set...')
        triplet_test(my_model, test_line_data_loader, train_flag=False, loss_function=loss_function, y_true=y_true, y_pred=y_pred)    
        for index1 , i2 in enumerate((0, 0.5, 1, 1.5, 2)):
            print(f"test_acc{i2}: ",100* (sum(acc_for_batches[index1]) /  len(acc_for_batches[index1])) )
            cf_matrix = confusion_matrix(y_true[index1], y_pred[index1])
            classes = ('0', '1')
            df_cm = pd.DataFrame(cf_matrix / 8000, index = [i for i in classes],
                        columns = [i for i in classes])
            plt.figure(figsize = (12,7))
            writer.add_figure('confusion_matrix_thresh_{}'.format(i2), sn.heatmap(df_cm, annot=True).get_figure(), i)

        writer.add_scalars("test_acc",{str(value) : (100*(sum(acc_for_batches[indx]) /  len(acc_for_batches[indx]))) for indx, value in enumerate((0, 0.5, 1, 1.5, 2))},i)

        writer.add_scalars("losses",{"train_loss": (sum(loss_history_for_epoch) / len(loss_history_for_epoch)), 'test_loss': (sum(loss_history_for_epoch_test) / len(loss_history_for_epoch_test))},i)


        
    writer.close()

