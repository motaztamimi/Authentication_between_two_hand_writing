import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from data_set import LinesDataSet
from way_2_model import Net
import numpy as np
import seaborn as sn
import pandas as pd
import sys


def train ( model, loss_function, optimizer,train_loader,loss_history, epoch):
    model.train()
    batches_loss = []
    for _ , data in enumerate(train_loader):
        image1, image2, label = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image1 = image1[:,None,:,:]
        image2 = image2[:,None,:,:]
        label = label.cuda()

        output = model(image1, image2)
        optimizer.zero_grad()
        new_label = torch.empty((label.shape[0], 2)).float().cuda()

        for i in range(label.shape[0]):
            if label[i].item() == 1.:
                new_label[i][0] = 0.
                new_label[i][1] = 1.
            if label[i].item() == 0.:
                new_label[i][0] = 1.
                new_label[i][1] = 0.

        loss = loss_function(output, new_label)
        loss_history.append(loss.cpu().item())
        batches_loss.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
    loss_history_for_ephoces.append( sum(batches_loss) / len(batches_loss))



def test ( model,test_loader, acc_history, train_flag):
    model.eval()
    for _ , data in enumerate(test_loader):
        image1, image2, label = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image1 = image1[:,None,:,:]
        image2 = image2[:,None,:,:]
        label = label.cuda()
        with torch.no_grad():
            output = model(image1, image2)
            predeict_= torch.argmax(output)
            y_pred.append(predeict_.cpu().item())
            y_true.append(label.cpu().item())
            if predeict_.item() == label.item():
                acc_history.append(1)
            else:
                acc_history.append(0)
    
    print('test acc: {}'.format(sum(acc_history) / len(acc_history)))
    if train_flag:
        all_acc_train.append(sum(acc_history) / len(acc_history))
    else:
        all_acc_test.append(sum(acc_history) / len(acc_history))




if __name__ == "__main__":

    train_line_data_set = LinesDataSet(csv_file="Train_labels_for_english.csv", root_dir="english_data_set", transform=transforms.Compose([transforms.ToTensor()]))
    test_line_data_set = LinesDataSet(csv_file="Test_labels_for_english.csv", root_dir='english_data_set', transform=transforms.Compose([transforms.ToTensor()]))
    train_line_data_loader = DataLoader(train_line_data_set,shuffle=True,batch_size=10)
    test_line_data_loader = DataLoader(test_line_data_set, shuffle=True, batch_size=1)
    train_line_data_loader_for_test = DataLoader(train_line_data_set,shuffle=True,batch_size=1)

    torch.manual_seed(17)

    loss_function = nn.MSELoss()

    loss_history = []
    loss_history_for_ephoces = []
    all_acc_test = []
    all_acc_train = []
    my_model = Net()
        

    my_model = my_model.cuda()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001, weight_decay=0.0001)

    # my_model.load_state_dict(torch.load('model_v2_lr_0,001_adam_outs_2_18layer_epchs_20_labels_10000_acc_70.pt', map_location='cuda:0'))
    epoches = 5
    for i in range(epoches):

        print('epoch number: {}'.format(i + 1))
        train(my_model, loss_function, optimizer, train_line_data_loader, loss_history, i + 1)
        print('epoch loss: {}'.format(loss_history_for_ephoces[i]))

        torch.save(my_model.state_dict(), 'model.pt')

        y_pred = []
        y_true = []

        print('Testing on Train Data_set...')
        test(my_model, train_line_data_loader_for_test, acc_history = [], train_flag = True)

        y_pred = []
        y_true = []

        print('Testing on Test Data_set...')
        test(my_model, test_line_data_loader, acc_history = [], train_flag = False)

        plt.subplot(131)
        plt.plot(loss_history_for_ephoces)
        plt.title('loss')

        plt.subplot(132)
        plt.plot(all_acc_test)
        plt.title('test acc')

        plt.subplot(133)
        plt.plot(all_acc_train)
        plt.title('train acc')

        plt.show()

        
   