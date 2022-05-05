import numpy as np
import torch
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models.triplet_model import ResNet, ResidualBlock
from torch.utils.data import DataLoader
from dataSets.data_set import LinesDataSetTripletWithLabel
import torchvision.transforms as transforms
import torch.nn.functional as F
def test(model, test_loader, thresh):
    model.eval()
    acc  = []
    for _, data in enumerate(test_loader):
        image1, image2, image3, label = data
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
            pred = (dist_a_n - dist_a_p - thresh).cpu().data
            pred = pred.reshape(pred.size()[0], 1)
            acc.append(( ((pred > 0 )[label == 1.0].sum() + (pred <= 0)[label == 0.0].sum())/ dist_a_p.size()[0]).data.item())

    
    print(f'acc with thresh ({i}): {100 * sum(acc) / len(acc)}')


if __name__ == '__main__':
    torch.manual_seed(17)
    my_model = ResNet(ResidualBlock, [2, 2, 2]).cuda()
    test_line_data_set = LinesDataSetTripletWithLabel(csv_file="../test_labels_for_arabic_triplet.csv", root_dir='../data_for_each_person',  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]))
    test_line_data_loader = DataLoader(test_line_data_set, shuffle=True, batch_size=20)
    my_model.load_state_dict(torch.load(r"../model_epoch_20.pt", map_location='cuda:0'))
    for i in np.linspace(5, 10, num=50, endpoint=True):
        test(my_model, test_line_data_loader, i)