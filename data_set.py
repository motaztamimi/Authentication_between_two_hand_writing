import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib.pyplot import figure, prism


class LinesDataSet(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):

            idx = idx.tolist()

        img0_path = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        img1_path = os.path.join(self.root_dir, self.labels.iloc[idx, 1])
        img0 = io.imread(img0_path)
        img1 = io.imread(img1_path)

        label = self.labels.iloc[idx, 2]

        if self.transform:
            self.transform(img0)
            self.transform(img1)

        return {'img0': img0, 'img1': img1, 'label': label}


line_dataSet = LinesDataSet(csv_file='Train_Labels.csv',
                            root_dir='data_for_each_person')

fig = plt.figure()
# figure(figsize=(4, 4), dpi=20)

for i in range(1, len(line_dataSet), 2):
    sample = line_dataSet[i]

    print(i, sample['img0'].shape, sample['img1'].shape, sample['label'])
    ax = plt.subplot(1, 2, i)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample['img0'], cmap='gray')

    ax = plt.subplot(1, 2, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample['img1'], cmap='gray')
    if i == 1:
        plt.show()
        break
