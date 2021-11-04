import glob
import os

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from configuration import Config


class MyDataset(Dataset):
    def __init__(self, config: Config, transform: transforms = None, mode='train'):
        self.files_list = list()  # file names
        self.labels = list()
        self.dir = config.dir
        self.kind_map = config.kind_map
        self.transforms = transform
        self.config = config
        self.mode = mode
        if mode == 'train':
            for kind in self.kind_map.keys():
                kind_file_list = glob.glob(self.dir + os.sep + kind + '*.jpg')
                kind_label = [self.kind_map[kind] for x in range(len(kind_file_list))]
                self.files_list.extend(kind_file_list)
                self.labels.extend(kind_label)
        elif mode == 'test':
            files_list = glob.glob(self.dir + os.sep + '*')
            for i in range(1, len(files_list) + 1):
                self.files_list.append(self.dir + os.sep + str(i) + '.jpg')
            self.ids = [x for x in range(len(self.files_list))]
        return

    def __getitem__(self, index):
        img = Image.open(self.files_list[index]).convert('RGB')

        if self.transforms is None:
            self.transforms = transforms.Compose([
                transforms.Resize(self.config.input_shape),
                transforms.ToTensor()
            ])
        img = self.transforms(img)

        X = img
        if self.mode == 'train':
            Y = self.labels[index]
        else:
            Y = self.ids[index]
        return X, Y

    def __len__(self):
        return len(self.files_list)


if __name__ == '__main__':
    config = Config()
    dataset = MyDataset(config=config)
    my_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.Resize(config.input_shape),
        transforms.ToTensor()
    ])
    full_dataset = MyDataset(config)
    train_len = int(config.split_rate * len(full_dataset))
    valid_len = len(full_dataset) - train_len
    train_set, valid_set = torch.utils.data.random_split(full_dataset, [train_len, valid_len])
    # train_set.dataset.transforms = my_transforms

    for x in range(10):
        img, label = train_set[0]
        print(img)
        img = transforms.ToPILImage()(img)
        print(img)
        plt.imshow(img)
        plt.show()
