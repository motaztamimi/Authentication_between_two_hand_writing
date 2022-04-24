import argparse
# from dataSets.data_set import LinesDataSet, LinesDataSetTriplet

# from dataManpiulation import create_excel_for_triplet_from_excel, create_labels_form_excel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', metavar='model', type=str, help='the model to be trained')
    parser.add_argument('--lr', metavar='lr', type=float, help='learing rate')
    parser.add_argument('--weigth-decay', metavar='weight-decay', type=float, help='weight decay')
    parser.add_argument('--lr-decay', metavar='lr-decay', type=bool, help='wheather to reduce learning rate each epoch or not')
    parser.add_argument('--epoch', metavar='epch', type=int, help='number of epochs to train the model')
    args = parser.parse_args()

    # model = customResNet().cuda()


    print(args)