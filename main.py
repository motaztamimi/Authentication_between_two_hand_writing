import argparse
from dataSets.data_set import LinesDataSet, LinesDataSetTriplet
from dataManpiulation import create_excel_for_triplet_from_excel, create_labels_form_excel
from models.customResNet import ResNet, ResidualBlock
from models import resnet18
from losses.losses import TripletLoss
import distutils.util
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', metavar='model', type=str, help='the model to be trained')
    parser.add_argument('--lr', metavar='lr', type=float, help='learing rate', default=0.001)
    parser.add_argument('--weigth_decay', metavar='weight-decay', type=float, help='weight decay', default=0.0001)
    parser.add_argument('--lr_decay', metavar='lr-decay', type=lambda x:bool(distutils.util.strtobool(x)), help='wheather to reduce learning rate each epoch or not', default=False)
    parser.add_argument('--epochs', metavar='epochs', type=int, help='number of epochs to train the model', default=30)
    parser.add_argument('--loss', metavar='loss', type=str,help='loss function', default='MSE')
    parser.add_argument('--cuda',metavar='cuda', type=lambda x:bool(distutils.util.strtobool(x)), help='wether to use gpu or not', default=True)
    args = parser.parse_args()

    model = ResNet(ResidualBlock, [2, 2, 2]) if args.model != 'resnet18' else resnet18()
    lr = args.lr
    weight_decay = args.weigth_decay
    epochs = args.epochs
    if args.cuda:
        print(args.cuda)
        model = model.cuda()
    

