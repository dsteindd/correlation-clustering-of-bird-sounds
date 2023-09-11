from __future__ import print_function

import argparse
import os
import shutil
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

import config
from torch_callbacks import CallbackContainer, CSVLogger, EarlyStoppingCallback, ModelCheckpoint, LRScheduleCallback, \
    MonitorMode
from torch_datasets import TargetEncoding, AudioDeterministicBirdDataset, AudioRandomBirdDataset, read_species_list, \
    RandomApply, RandomExclusiveListApply, ESC50NoiseInjection, Noise, NoiseType, RollAndWrap, RollDimension
from torch_metrics import MetricContainer, BinaryAccuracy, ConfusionMatrix, Recall, Precision


def build_embedding_network(
        num_classes: int,
        is_multilabel: bool = False
):
    embedding_network = torchvision.models.resnet18(num_classes=num_classes)

    # over-write the first conv layer to use gray-scale images
    embedding_network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    embedding_network = nn.Sequential(
        embedding_network,
        nn.Sigmoid() if is_multilabel else nn.Softmax(dim=1)
    )

    embedding_network.kwargs = {
        'num_classes': num_classes,
        'is_multilabel': is_multilabel
    }

    return embedding_network


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 callbacks: CallbackContainer = None,
                 metrics: MetricContainer = None,
                 log_interval=10,
                 dry_run=False
                 ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks or CallbackContainer()
        self.metrics = metrics or MetricContainer()
        self.log_interval = log_interval
        self.dry_run = dry_run

    def train(self,
              epochs: int,
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader = None
              ):

        for epoch in range(1, epochs + 1):
            logs = {

            }
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
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            self.metrics.update(outputs, targets)
            logs["loss"] += loss

            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           logs["loss"] / (batch_idx + 1)) + self.metrics.summary_string())
                if self.dry_run:
                    break
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
            batch_idx = 0
            for batch_idx, (images, targets) in enumerate(val_loader):
                images, targets = images.to(self.device), targets.to(
                    self.device)
                outputs = self.model(images)
                logs["val_loss"] += self.criterion(outputs, targets).sum().item()  # sum up batch loss
                self.metrics.update(outputs, targets)

        logs["val_loss"] /= (batch_idx + 1)

        print('\nTest set: Average loss: {:.4f}\t'.format(logs["val_loss"]) + self.metrics.summary_string() + "\n")

        logs.update(self.metrics.summary())
        self.metrics.reset()

        return logs


def get_metrics() -> MetricContainer:
    return MetricContainer([
        BinaryAccuracy(threshold=0.5, name="accuracy", precision=3),
        ConfusionMatrix(precision=3),
        Recall(precision=3),
        Precision(precision=3)
    ])


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Classification Networks Example')
    parser.add_argument('--data-dirs',
                        nargs='+',
                        default=["./data"],
                        help="Data directories for input files"
                        )
    parser.add_argument('--val-data-dirs',
                        nargs='+',
                        default=None,
                        help="Optional validation data directory"
                        )
    parser.add_argument('--species-list',
                        type=str,
                        default=None,
                        help="Where to find a list of species. "
                             "Line number will be used as index in the predictions (default: species-3600.txt)"
                        )
    parser.add_argument('--esc-50-dir',
                        default=config.ESC_DIR,
                        type=str,
                        help="Where to find the ESC-50 dataset"
                        )
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-name',
                        type=str,
                        help='model name',
                        default="test-model-torch"
                        )
    parser.add_argument('--multilabel',
                        default=False,
                        help="Whether to train the network as multilabel or not",
                        action='store_true'
                        )
    parser.add_argument('--min-sns',
                        type=int,
                        default=None,
                        help="Filter files with sns (encoded in file name) with SNS < MIN_SNS (default: None)"
                        )
    parser.add_argument('--override',
                        help='Whether to override models under the same name',
                        action='store_true',
                        default=False
                        )
    parser.add_argument('--model-dir',
                        type=str,
                        default=config.MODELS_DIR,
                        help="Where to save checkpoints and such."
                        )
    parser.add_argument('--augment',
                        default=False,
                        action='store_true',
                        help="Whether to use augmentation or not (default: False)"
                        )
    args = parser.parse_args()

    if args.val_data_dirs is None:
        args.val_data_dirs = args.data_dirs

    model_dir = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "siamese_network.pt")

    if os.path.exists(model_path) and not args.override:
        print("Model exists. Specify --override to override")
        exit()
    if os.path.exists(model_path) and args.override:
        shutil.rmtree(model_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        print("Using CUDA device...")
        device = torch.device("cuda")
    else:
        print("Using CPU...")
        device = torch.device("cpu")

    train_kwargs = {
        'batch_size': args.batch_size
    }
    test_kwargs = {
        'batch_size': args.test_batch_size
    }
    if use_cuda:
        num_workers = len(os.sched_getaffinity(0))
        print(f"Training with {num_workers} workers...")
        cuda_kwargs = {
            'num_workers': num_workers,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.augment:
        audio_transforms = torch.nn.Sequential(
            RandomApply(
                module=RandomExclusiveListApply(
                    choice_modules=torch.nn.ModuleList([
                        ESC50NoiseInjection(
                            dst_sr=config.SAMPLE_RATE,
                            directory=args.esc_50_dir,
                            min_snr_db=0.0,
                            max_snr_db=30.0
                        ),
                        Noise(scale=NoiseType.WHITE, min_snr_db=0.0, max_snr_db=30.0),
                        Noise(scale=NoiseType.PINK, min_snr_db=0.0, max_snr_db=30.0)
                    ]),
                ),
                p=0.4
            ),
        ),

        mel_transforms = torch.nn.Sequential(
            RandomApply(
                module=torch.nn.Sequential(
                    torchaudio.transforms.FrequencyMasking(freq_mask_param=20 if args.nmels == 128 else 10),
                    torchaudio.transforms.FrequencyMasking(freq_mask_param=20 if args.nmels == 128 else 10)
                ),
                p=0.4
            ),
            RandomApply(
                module=RollAndWrap(max_shift=5, min_shift=-5, dim=RollDimension.FREQUENCY),
                p=0.4
            ),
            RandomApply(
                module=RollAndWrap(max_shift=int(44100 / (args.nfft // 4) * 0.4),
                                   min_shift=int(-44100 / (args.nfft // 4) * 0.4),
                                   dim=RollDimension.TIME),
                p=0.4
            ),
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        )
    else:
        audio_transforms = None
        mel_transforms = torch.nn.Sequential(
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        )

    train_dataset = AudioRandomBirdDataset(
        data_dirs=args.data_dirs,
        split="train",
        nfft=config.NFFT,
        n_mels=config.NMELS,
        min_sns=args.min_sns,
        target_encoding=TargetEncoding.ONE_HOT,
        species_list=read_species_list(args.species_list),
        audio_transforms=audio_transforms,
        mel_transforms=mel_transforms
    )

    val_dataset = AudioDeterministicBirdDataset(
        args.val_data_dirs,
        split="val",
        nfft=config.NFFT,
        n_mels=config.NMELS,
        min_sns=args.min_sns,
        species_list=read_species_list(args.species_list),
        mel_transforms=torch.nn.Sequential(
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        ))

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    # model building phase
    model = build_embedding_network(
        num_classes=len(train_dataset.species),
        is_multilabel=args.multilabel
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.BCELoss(),
        device=device,
        metrics=MetricContainer([
            BinaryAccuracy(threshold=0.5, name="accuracy", precision=3),
            ConfusionMatrix(precision=3),
            Recall(precision=3),
            Precision(precision=3)
        ]),
        callbacks=CallbackContainer([
            CSVLogger(os.path.join(model_dir, "history.csv")),
            EarlyStoppingCallback(monitor="val_loss", patience=100),
            ModelCheckpoint(
                models_dict={
                    "siamese.pt": model
                },
                checkpoint_dir=os.path.join(model_dir, "val_loss_checkpoints")
            ),
            ModelCheckpoint(
                monitor='val_accuracy',
                mode=MonitorMode.MAX,
                models_dict={
                    "siamese.pt": model
                },
                checkpoint_dir=os.path.join(model_dir, "val_accuracy_checkpoints")
            ),
            LRScheduleCallback(
                ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5
                ),
                monitor="loss"
            )
        ])
    )

    trainer.train(args.epochs, train_loader, val_loader)

    if not os.path.exists(model_path):
        torch.save([model.kwargs, model.state_dict()], model_path)


if __name__ == '__main__':
    main()
