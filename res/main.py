import os

import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from MyDataset import MyDataset
from MyNet import MyNet
from configuration import Config
from engine import train_one_epoch, evaluate


# os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def prepare_data(config: Config):
    my_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation((0, 90)),
        transforms.Resize(config.input_shape),
        transforms.ToTensor()
    ])
    full_dataset = MyDataset(config)
    train_len = int(config.split_rate * len(full_dataset))
    valid_len = len(full_dataset) - train_len
    train_set, valid_set = torch.utils.data.random_split(full_dataset, [train_len, valid_len])
    train_set.dataset.transforms = my_transforms
    train_dataloader = DataLoader(train_set, config.batch_size, True, num_workers=config.num_workers)
    valid_dataloader = DataLoader(valid_set, config.batch_size, False, num_workers=config.num_workers)

    return train_dataloader, valid_dataloader


def main(config: Config):
    train_dataloader, valid_dataloader = prepare_data(config=config)

    model = MyNet(config).to(config.device)

    if os.path.exists(config.checkpoint):
        print(f'loading check point from: {config.checkpoint}...')
        checkpoint = torch.load(config.checkpoint)
        model = checkpoint['model']
        config.start_epoch = checkpoint['epoch']

    print('start training...')

    for epoch in range(config.start_epoch + 1, config.epoch + 1):
        loss, acc = train_one_epoch(model, model.criterion, train_dataloader, model.optimizer, config.device)
        print(f'[train loss {epoch}/{config.epoch}]: {loss:.5f}')
        print(f'[train acc {epoch}/{config.epoch}]: {acc:.5f}')

        loss, acc = evaluate(model, model.criterion, valid_dataloader, config.device)
        print(f'[evaluate loss {epoch}/{config.epoch}]: {loss:.5f}')
        print(f'[evaluate acc {epoch}/{config.epoch}]: {acc:.5f}')

        print(f'saving checkpoint at {config.checkpoint}...')
        torch.save({
            'model': model,
            'epoch': epoch
        }, config.checkpoint)


if __name__ == '__main__':
    config = Config()
    config.epoch = 10
    main(config)
