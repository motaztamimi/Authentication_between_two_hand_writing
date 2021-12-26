import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

from data_set import LinesDataSet

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = torchvision.models.resnet18(pretrained = False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, 3,bias=False)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(nn.Linear(num_features, 200), nn.ReLU())
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
        print("epoch Number: {} bathc_loss: {}".format(epoch, loss.item()))
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
    torch.manual_seed(17)
    train_line_data_set = LinesDataSet(csv_file="Train_Labels.csv", root_dir="data_for_each_person", transform=transforms.Compose([transforms.ToTensor()]))
    test_line_data_set = LinesDataSet(csv_file="Test_Labels.csv", root_dir='data_for_each_person', transform=transforms.Compose([transforms.ToTensor()]))
    train_line_data_loader = DataLoader(train_line_data_set,shuffle=True,batch_size=17)
    test_line_data_loader = DataLoader(test_line_data_set, shuffle=True, batch_size=1)
    train_line_data_loader_for_test = DataLoader(train_line_data_set,shuffle=True,batch_size=1)

    loss_function = nn.MSELoss()

    loss_history = []
    loss_history_for_ephoces = []
    all_acc_test = []
    all_acc_train = []
    my_model = Net().cuda()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)

    # my_model.load_state_dict(torch.load('model.pt', map_location='cuda:0'))
    epoches = 30
    for i in range(epoches):
        train(my_model, loss_function, optimizer, train_line_data_loader, loss_history, i + 1)
        torch.save(my_model.state_dict(), 'model.pt')
        print('Testing on Train Data_set...')
        test(my_model, train_line_data_loader_for_test, acc_history = [], train_flag = True)
        print('Testing on Test Data_set...')
        test(my_model, test_line_data_loader, acc_history = [], train_flag = False)


    plt.subplot(2,2,1)
    plt.plot(loss_history_for_ephoces)
  
    plt.title('train epoches loss')
    plt.subplot(2,2,2)
    plt.plot(loss_history)

    plt.title('train loss')
    plt.subplot(2,2,3)
    plt.plot(all_acc_train)
 
    plt.title('train acc')
    plt.subplot(2,2,4)
    plt.plot(all_acc_test)
    plt.title('test acc')
    plt.savefig("model result.jpg")
    plt.show()

