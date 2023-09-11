import torch
from torch_metrics import MetricContainer
from torch import nn
from torch_callbacks import CallbackContainer
from typing import Dict


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 callbacks: CallbackContainer = None,
                 metrics: MetricContainer = None,
                 log_interval=10
                 ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks or CallbackContainer()
        self.metrics = metrics or MetricContainer()
        self.log_interval = log_interval

    def train(self,
              epochs: int,
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader = None
              ):

        for epoch in range(1, epochs + 1):
            logs = {

            }
            self.callbacks.on_epoch_begin(epoch, logs)
            train_logs = self.train_epoch(epoch, train_loader)
            val_logs = self.val_epoch(epoch, val_loader)
            logs.update(train_logs)
            logs.update(val_logs)
            logs = self.callbacks.on_epoch_end(epoch, logs)
            if "stop_training" in logs.keys() and logs["stop_training"]:
                break

    def train_epoch(self, epoch, train_loader: torch.utils.data.DataLoader):
        self.model.train()
        self.metrics.set_train()

        logs = {
            "epoch": epoch,
            "loss": 0
        }

        batch_idx = 0
        for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
            images_1, images_2, targets = images_1.to(self.device), images_2.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images_1, images_2)

            self.metrics.update(outputs, targets)

            loss = self.criterion(outputs, targets)
            logs["loss"] += loss
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(images_1), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    logs["loss"] / (batch_idx + 1)) + self.metrics.summary_string())

        logs["loss"] /= (batch_idx + 1)

        logs.update(self.metrics.summary())
        self.metrics.reset()
        return logs

    def val_epoch(self, epoch, val_loader: torch.utils.data.DataLoader) -> Dict:
        self.model.eval()
        self.metrics.set_val()
        logs = {
            "val_loss": 0,
            "epoch": epoch
        }

        with torch.no_grad():
            for batch_idx, (images_1, images_2, targets) in enumerate(val_loader):
                images_1, images_2, targets = images_1.to(self.device), images_2.to(self.device), targets.to(
                    self.device)
                outputs = self.model(images_1, images_2)
                logs["val_loss"] += self.criterion(outputs, targets).sum().item()  # sum up batch loss
                self.metrics.update(outputs, targets)

                if batch_idx % self.log_interval == 0:
                    print('Val Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
                        epoch, batch_idx * len(images_1), len(val_loader.dataset),
                               100. * batch_idx / len(val_loader)))

        logs["val_loss"] /= (batch_idx + 1)

        print('\nTest set: Average loss: {:.4f}\t'.format(logs["val_loss"]) + self.metrics.summary_string() + "\n")

        logs.update(self.metrics.summary())
        self.metrics.reset()

        return logs
