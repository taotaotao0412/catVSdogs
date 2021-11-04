import argparse
import glob
import random
import matplotlib
from matplotlib import pyplot as plt
import torch

import engine
from configuration import Config
from torchvision.transforms import transforms
from PIL import Image
import os


@torch.no_grad()
def predict_one_picture(model, img: Image, config: Config):
    model.eval()
    model.to(config.device)
    model.model.to(config.device)
    trans = transforms.Compose([
        transforms.Resize(config.input_shape),
        transforms.ToTensor()
    ])
    img = trans(img)
    img = torch.unsqueeze(img, dim=0)

    logistic = model(img.to(config.device))
    kind = logistic.argmax(dim=-1).cpu()
    return kind.item()


def plot_imgs(images, labels, args):
    kind_map = {0: 'Dog', 1: 'Cat'}
    row = args.row
    if row is None:
        row = int(args.n**0.5 + 1)
    col = int(len(images) / row)
    if row * col < len(images):
        col += 1
    for i in range(0, len(images)):
        plt.subplot(row, col, i + 1)
        plt.imshow(images[i])
        if labels is None:
            plt.title('???')
        else:
            plt.title(kind_map[labels[i]])
        plt.axis('off')
    plt.subplots_adjust(hspace=0.2)
    plt.show()


def main(args, config: Config):
    if args.n is None:
        args.n = 9
    model = torch.load(config.checkpoint)['model'].to(config.device)

    imgs_paths = glob.glob('../dataset/train/*')
    imgs_paths = random.sample(imgs_paths, args.n)

    imgs_list = list()
    for path in imgs_paths:
        imgs_list.append(Image.open(path))
    labels = list()
    for image in imgs_list:
        labels.append(predict_one_picture(model, image, config))
    plot_imgs(imgs_list, labels, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict a photo is a dog or cat')
    parser.add_argument('--file', type=str, required=False)
    parser.add_argument('-n', type=int, required=False)
    parser.add_argument('--dir', type=str, required=False)
    parser.add_argument('--row', type=int, required=False)
    args = parser.parse_args()

    config = Config()
    main(args, config)
