from __future__ import print_function

import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchaudio

import config
from torch_callbacks import CallbackContainer, TimeCallback, CSVLogger, EarlyStoppingCallback, ModelCheckpoint, \
    MonitorMode
from torch_datasets import AudioBirdSongPairs, read_species_list
from torch_metrics import BinaryAccuracy, MetricContainer, ConfusionMatrix
from torch_siamese_trainer import Trainer
from torch_datasets import RandomApply, RandomExclusiveListApply, Noise, ESC50NoiseInjection, RollAndWrap, RollDimension, NoiseType


class Concatenation(torch.nn.Module):
    def __init__(self):
        super(Concatenation, self).__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], 1)

    def __call__(self, x1, x2, *args, **kwargs):
        return self.forward(x1, x2)


def build_resnet18(
        embedding_dimension: int
):
    embedding_network = torchvision.models.resnet18()

    # over-write the first conv layer to use gray-scale images
    embedding_network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    fc_in_features = embedding_network.fc.in_features
    embedding_network = nn.Sequential(*(list(embedding_network.children())[:-1]))
    embedding_network = nn.Sequential(
        embedding_network,
        torch.nn.Flatten(),
        torch.nn.Linear(fc_in_features, embedding_dimension)
    )

    return embedding_network


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer.
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """

    def __init__(self,
                 embedding_dimension: int,
                 ):
        super(SiameseNetwork, self).__init__()

        self.kwargs = {
            'embedding_dimension': embedding_dimension
        }

        self.embedding = build_resnet18(embedding_dimension)

        # initialize the weights
        self.embedding.apply(self.init_weights)

        self.fc = None
        self.merge_layer = None
        self.sigmoid = None

        # merge layer
        self.merge_layer = Concatenation()
        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dimension, embedding_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dimension, 1),
            nn.Flatten()
        )
        self.sigmoid = nn.Sigmoid()

        if self.fc:
            self.fc.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.embedding(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if self.merge_layer:
            outputs = self.merge_layer(output1, output2)

            if self.fc:
                outputs = self.fc.forward(outputs)

            if self.sigmoid:
                outputs = self.sigmoid(outputs)

            return outputs.view(outputs.size()[0])

        return output1, output2

    @property
    def embedding_network(self):
        return self.embedding

    @property
    def args(self):
        return self.kwargs


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Siamese Networks for Bird Sound Clustering')
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
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1024,
                        metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs',
                        type=int,
                        default=150,
                        metavar='N',
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval',
                        type=int,
                        default=10,
                        metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-name',
                        type=str,
                        help='model name',
                        default="test-model-torch"
                        )
    parser.add_argument('--override',
                        help='Whether to override models under the same name',
                        action='store_true',
                        default=False
                        )
    parser.add_argument('--embedding-dimension',
                        type=int,
                        default=128
                        )
    parser.add_argument('--min-sns',
                        type=int,
                        default=config.MIN_SNS,
                        help="Filter files with sns (encoded in file name) with SNS < MIN_SNS (default: None)"
                        )
    parser.add_argument('--model-dir',
                        type=str,
                        default=config.MODELS_DIR,
                        help="Where to save checkpoints and such."
                        )
    parser.add_argument('--esc-50-dir',
                        default=config.ESC_DIR,
                        type=str,
                        help="Where to find the ESC-50 data"
                        )
    parser.add_argument('--augment',
                        default=False,
                        action='store_true',
                        help="Whether to use augmentation or not. (default: False)"
                        )
    args = parser.parse_args()

    if args.val_data_dirs is None:
        args.val_data_dirs = args.data_dirs

    model_dir = os.path.join(args.model_dir, args.model_name)
    if args.override and os.path.exists(model_dir):
        shutil.rmtree(model_dir)

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
        print(f"Running with {num_workers} workers...")
        cuda_kwargs = {
            'num_workers': num_workers,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.augment:
        audio_transforms=torch.nn.Sequential(
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
                    torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
                    torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
                ),
                p=0.4
            ),
            RandomApply(
                module=RollAndWrap(max_shift=5, min_shift=-5, dim=RollDimension.FREQUENCY),
                p=0.4
            ),
            RandomApply(
                module=RollAndWrap(max_shift=int(config.SAMPLE_RATE / (config.NFFT // 4) * 0.4),
                                   min_shift=int(-config.SAMPLE_RATE / (config.NFFT // 4) * 0.4),
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

    train_dataset = AudioBirdSongPairs(
        args.data_dirs,
        split="train",
        min_sns=args.min_sns,
        nfft=config.NFFT,
        n_mels=config.NMELS,
        species_list=read_species_list(args.species_list),
        audio_transforms=audio_transforms,
        mel_transforms=mel_transforms
    )
    # no transforms here
    test_dataset = AudioBirdSongPairs(
        args.val_data_dirs,
        split="val",
        min_sns=args.min_sns,
        min_k_pair=32,
        nfft=config.NFFT,
        n_mels=config.NMELS,
        species_list=read_species_list(args.species_list),
        mel_transforms=torch.nn.Sequential(
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        ))

    print("Train Set Size: ", len(train_dataset))
    print("Val Set Size: ", len(test_dataset))
    print(f"Training on {len(train_dataset.species)} bird species. Validating on {len(test_dataset.species)} bird species.")

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # model building phase
    model = SiameseNetwork(embedding_dimension=args.embedding_dimension).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.BCELoss(),
        device=device,
        metrics=MetricContainer([
            BinaryAccuracy(name='accuracy', precision=3),
            ConfusionMatrix(pos_min=0.5, pos_max=1.0, precision=0)
        ]),
        callbacks=CallbackContainer([
            TimeCallback(),
            CSVLogger(os.path.join(model_dir, "history.csv")),
            EarlyStoppingCallback(monitor="val_accuracy", patience=300, mode=MonitorMode.MAX),
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
            )
        ])
    )

    trainer.train(args.epochs, train_loader, val_loader)

    if not os.path.exists(model_path):
        torch.save([model.kwargs, model.state_dict()], model_path)


if __name__ == '__main__':
    main()
