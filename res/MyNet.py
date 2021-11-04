import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.auto import tqdm

from configuration import Config


class MyNet(nn.Module):
    def __init__(self, config: Config, pretrain=True):
        super().__init__()
        self.config = config
        res_net = models.resnet18(pretrained=pretrain).to(config.device)
        self.model = res_net
        fc_features = self.model.fc.in_features
        fc = nn.Sequential(
            nn.Linear(in_features=fc_features, out_features=512),
            nn.Linear(in_features=512, out_features=512),
            nn.Linear(in_features=512, out_features=len(config.kind_map.keys())),
        )
        self.model.fc = fc

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), config.lr, weight_decay=config.weight_decay)
        return

    def forward(self, x):
        x = self.model(x)
        return x

    def train_data(self, train_dataloader: DataLoader, valid_dataloader=None):
        self.train()

        for epoch in range(self.config.epoch):
            train_loss = list()
            train_accs = list()
            for batch in tqdm(train_dataloader):
                imgs, labels = batch
                logits = self.model(imgs.to(self.config.device))
                loss = self.criterion(logits, labels.to(self.config.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                acc = (logits.argmax(dim=-1) == labels.to(self.config.device)).float().mean()
                train_loss.append(loss.item())
                train_accs.append(acc)
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            print(f"[ Train | {epoch + 1:03d}/{self.config.epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

            # valid data:
            if valid_dataloader is not None:
                self.valid_data(valid_dataloader=valid_dataloader)

    def valid_data(self, valid_dataloader: DataLoader):
        self.eval()
        valid_loss = []
        valid_accs = []

        for batch in tqdm(valid_dataloader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = self.model(imgs.to(self.config.device))

            # We can still compute the loss (but not the gradient).
            loss = self.criterion(logits, labels.to(self.config.device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(self.config.device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


if __name__ == '__main__':
    None

