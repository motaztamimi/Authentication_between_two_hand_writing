from way_2_model import Net
import torch
import torchvision.transforms as transforms
from data_set import LinesDataSet
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


def test ( model,test_loader, acc_history, train_flag):
    model.eval()
    for _ , data in enumerate(test_loader):
        image1, image2, label = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image1 = image1[:,None,:,:]
        image2 = image2[:,None,:,:]
        label = label.cuda()
        
        # print(label)
        with torch.no_grad():
            output = model(image1, image2)
            # print(output)
            _, predeict_= torch.max(output, dim=1, keepdim=False)
            # print(predeict_)
            # print(predeict_.tolist().count(int(label[0])))
            if predeict_.tolist().count(int(label[1])) > 4:
                acc_history.append(1)
            else:
                acc_history.append(0)




if __name__ == '__main__':
    y_pred = []
    y_true = []
    acc_history = []
    test_data_set = LinesDataSet('test_labels_try.csv', 'data_for_each_person', transform=transforms.Compose([transforms.ToTensor()]))
    test_line_data_loader = DataLoader(test_data_set, shuffle=False, batch_size=8)
    model = Net()
    model = model.cuda()
    model.load_state_dict(torch.load('model_0.pt', map_location='cuda:0'))
    test(model, test_line_data_loader, acc_history=acc_history, train_flag=False)
    print(sum(acc_history) / len(acc_history))