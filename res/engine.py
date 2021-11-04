import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_one_epoch(model: nn.Module,
                    criterion: nn.CrossEntropyLoss,
                    data_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: str,
                    max_norm=None):
    model.train()
    criterion.train()
    epoch_loss = 0.
    epoch_acc = 0.
    total = len(data_loader)
    for batch in tqdm(data_loader):
        images, labels = batch
        logistic = model(images.to(device))
        loss = criterion(logistic, labels.to(device))
        acc = (logistic.argmax(dim=-1) == labels.to(device)).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        epoch_loss += loss_value
        epoch_acc += acc
    return epoch_loss / total, epoch_acc / total


@torch.no_grad()
def evaluate(model: nn.Module,
             criterion: nn.CrossEntropyLoss,
             data_loader: DataLoader,
             device: str
             ):
    model.eval()
    criterion.eval()

    total = len(data_loader)
    epoch_loss = 0.
    epoch_acc = 0.
    for batch in tqdm(data_loader):
        images, labels = batch
        logistic = model(images.to(device))
        loss = criterion(logistic, labels.to(device))
        acc = (logistic.argmax(dim=-1) == labels.to(device)).float().mean()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / total, epoch_acc / total
