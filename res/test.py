"""
    the code is to generate the result of Kaggle test data:https://www.kaggle.com/c/dogs-vs-cats/
"""

from glob import glob
import torch
from configuration import Config
from MyDataset import MyDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd


def main(config: Config):
    model = torch.load(config.checkpoint)['model'].to(config.device)

    config.dir = '../dataset/test1'
    test_dataset = MyDataset(config, mode='test')
    test_dataloader = DataLoader(test_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers)
    model.eval()
    res = list()
    for batch in tqdm(test_dataloader):
        imgs, _ = batch
        with torch.no_grad():
            logistic = model(imgs.to(config.device))
            res.append(logistic.argmax(dim=-1).cpu())

    with open('res.txt', 'w') as file:
        file.writelines('id,label')
        for i in range(len(res)):
            file.writelines(f'{i},{res[i]}')
    # print(res)
    print('write done')


if __name__ == '__main__':
    config = Config()
    #main(config)

    model = torch.load(config.checkpoint)['model'].to(config.device)

    config.dir = '../dataset/test1'
    test_dataset = MyDataset(config, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size,
                                 shuffle=False, num_workers=config.num_workers)
    model.eval()
    res = list()
    for batch in tqdm(test_dataloader):
        imgs, ids = batch
        with torch.no_grad():
            logistic = model(imgs.to(config.device))
            res.extend(torch.Tensor.tolist(logistic.argmax(dim=-1).cpu()))

    with open('res.txt', 'w') as file:
        file.writelines('id,label\n')
        for i in range(len(res)):
            file.write(f'{i + 1},{res[i]}\n')
    # print(res)
    print('write done')
