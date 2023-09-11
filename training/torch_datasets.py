import csv
import os
import random
from collections import defaultdict
from enum import Enum
from typing import Union, List

import numpy as np
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset


class TargetEncoding(Enum):
    ONE_HOT = 0
    CATEGORICAL = 1


class NoiseType(Enum):
    WHITE = 0
    PINK = 1
    BROWNIAN = 2
    BLUE = -1
    VIOLET = -2


class RollDimension(Enum):
    TIME = 2
    FREQUENCY = 1


class MinMaxScaling(torch.nn.Module):
    def __init__(self, new_min: 0, new_max: 1):
        super(MinMaxScaling, self).__init__()
        self.new_min = new_min
        self.new_max = new_max

    def forward(self, mel):
        if mel.min() != mel.max():
            return self.new_min + (mel - mel.min()) / (mel.max() - mel.min()) * (self.new_max - self.new_min)


class RollAndWrap(torch.nn.Module):
    def __init__(self,
                 max_shift: int,
                 dim: Union[int, RollDimension],
                 min_shift: int = None
                 ):
        super(RollAndWrap, self).__init__()
        self.max_shift = max_shift
        if min_shift is None:
            self.min_shift = -max_shift
        else:
            self.min_shift = min_shift

        if isinstance(dim, RollDimension):
            self.dim = dim.value
        else:
            self.dim = dim

    def forward(self, tensor):
        # draw random shift parameter
        shift = random.randint(self.min_shift, self.max_shift)
        return torch.roll(tensor, shifts=shift, dims=self.dim)


class RandomApply(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, p: float = 0.5):
        super(RandomApply, self).__init__()
        self.module = module
        self.p = p

    def forward(self, tensor):
        if torch.rand(1) <= self.p:
            return self.module.forward(tensor)

        return tensor


ESC50_BACKGROUND_ENVIRONMENTAL_CLASSES_AUGMENTS = [
    "rain",
    "water_drops",
    "wind",
    "pouring_water",
    "sea_waves",
    "thunderstorm",
    "crickets",
    "crackling_fire",
    "insects",
    "frog"
]

ESC50_BACKGROUND_ENVIRONMENTAL_CLASSES_ALL = [
    "dog", "rain", "crying_baby", "door_wood_knock", "helicopter",
    "rooster", "sea_waves", "sneezing", "mouse_click", "chainsaw",
    "pig", "crackling_fire", "clapping", "keyboard_typing", "siren",
    "cow", "crickets", "breathing", "door_wood_creaks", "car_horn",
    "frog", "coughing", "can_opening", "engine", "cat", "water_drops", "footsteps", "washing_machine", "train",
    "hen", "wind", "laughing", "vacuum_cleaner", "church_bells",
    "insects", "pouring_water", "brushing_teeth","clock_alarm", "airplane",
    "sheep", "toilet_flush", "snoring", "clock_tick", "fireworks",
    "crow", "thunderstorm", "drinking_sipping", "glass_breaking", "hand_saw"
]

class ESC50NoiseInjection(torch.nn.Module):
    def __init__(self,
                 directory: str,
                 dst_sr: int,
                 classes: List[str] = None,
                 min_snr_db: float = 1,
                 max_snr_db: float = 30,
                 probabilities: np.ndarray = None
                 ):
        super(ESC50NoiseInjection, self).__init__()
        classes = classes or ESC50_BACKGROUND_ENVIRONMENTAL_CLASSES_AUGMENTS

        self.classes = classes
        self.dst_sr = dst_sr
        if probabilities is not None:
            self.probabilities = probabilities / np.sum(probabilities)
        else:
            self.probabilities = np.ones(len(classes)) / len(classes)

        self.min_snr = min_snr_db
        self.max_snr = max_snr_db

        meta_path = os.path.join(directory, "meta", "esc50.csv")
        audio_dir = os.path.join(directory, "audio")

        self.grouped_examples = defaultdict(lambda: [])

        with open(meta_path, "r") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                category = row["category"]
                if category in classes:
                    self.grouped_examples[category].append(os.path.join(audio_dir, row["filename"]))

        missing_classes = set(classes).difference(set(self.grouped_examples.keys()))

        if len(missing_classes) != 0:
            print(f"Classes '{str.join(', ', missing_classes)}' were not found in the dataset")

    def forward(self, tensor):
        # choose random noise audio for augmentation
        random_class = np.random.choice(self.classes, p=self.probabilities)
        random_index = random.randint(0, len(self.grouped_examples[random_class]) - 1)
        path_to_noise_audio = self.grouped_examples[random_class][random_index]

        # read audio
        audio, sr = torchaudio.load(path_to_noise_audio)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.dst_sr)

        # pick a random start index
        # tensor (input audio) does have channels
        if tensor.shape[1] < audio.shape[1]:
            # input audio is shorter
            random_index = random.randint(0, audio.shape[1] - tensor.shape[1])
            audio = audio[:, random_index:random_index + tensor.shape[1]]
        elif tensor.shape[1] == audio.shape[1]:
            pass
        else:
            # tensor.shape[1] > audio.shape[1]
            # input audio is longer than noise audio
            num_repetitions = tensor.shape[1] // audio.shape[1]
            rest = tensor.shape[1] - num_repetitions * audio.shape[1]
            return_audio = torch.empty(tensor.shape[1])
            for j in range(num_repetitions):
                return_audio[j * audio.shape[1]:(j + 1) * audio.shape[1]] = audio
            return_audio[num_repetitions * audio.shape[1]:num_repetitions * audio.shape[1] + rest] = audio[:rest]
            audio = return_audio

        snr = random.uniform(self.min_snr, self.max_snr)
        tensor_squared = tensor.square().sum()
        audio_squared = audio.square().sum()

        # for testing noise signals only
        # otherwise the signal strength is zero, so the noise signal also gets scaled down to 0
        if tensor_squared == 0:
            return audio
        if audio_squared == 0:
            # this prevents divide by zero
            # for example in the ESC-50 dataset there are some 1s chunks which contain only zeros
            return tensor
        return tensor + torch.sqrt(10 ** (-snr / 10) * tensor_squared / audio_squared) * audio


class RandomExclusiveListApply(torch.nn.Module):
    """
    Applies exactly one of the transformations with the given probablities
    """

    def __init__(self, choice_modules: torch.nn.ModuleList, probabilities: np.ndarray = None):
        super(RandomExclusiveListApply, self).__init__()
        self.choice_modules = choice_modules
        if probabilities:
            self.probabilities = torch.tensor(probabilities / np.sum(probabilities))
        else:
            self.probabilities = torch.tensor(np.ones(len(choice_modules)) / len(choice_modules))

    def forward(self, tensor):
        if len(self.choice_modules) == 0:
            return tensor
        # todo: check if multithreading here is a problem
        module_index = torch.multinomial(self.probabilities, num_samples=1)
        # module_index = np.random.choice(range(len(self.choice_modules)), p=self.probabilities)
        return self.choice_modules[module_index].forward(tensor)

class Noise(torch.nn.Module):
    def __init__(self, scale: Union[int, NoiseType] = 0, min_snr_db=3.0, max_snr_db=30.0):
        """
        :param scale: Exponent of noise (default: 0)
        :param min_snr_db: Minimum signal-to-noise ratio in absolute values, i.e. not dB scale (SNR_DB = -10*log(snr))
        :param max_snr_db: Maximum signal-to-noise ratio in absolute values, i.e. not dB scale (SNR_DB = -10*log(snr))
        """
        super().__init__()
        if isinstance(scale, int):
            self.exponent = scale
        elif isinstance(scale, NoiseType):
            self.exponent = scale.value
        self.min_snr = min_snr_db
        self.max_snr = max_snr_db

    def psd(self, f):
        return 1 / np.where(f == 0, float('inf'), np.power(f, self.exponent / 2))

    def forward(self, audio):
        snr = random.uniform(self.min_snr, self.max_snr)

        # check if this is fast enough, or if there are functions from torch
        white_noise = np.fft.rfft(np.random.randn(audio.shape[1]))
        spectral_density = self.psd(np.fft.rfftfreq(audio.shape[1]))
        # Normalize S
        spectral_density = spectral_density / np.sqrt(np.mean(spectral_density ** 2))
        colored_noise = white_noise * spectral_density
        colored_noise = np.fft.irfft(colored_noise)
        colored_noise = torch.tensor(colored_noise, dtype=audio.dtype)
        colored_noise = colored_noise.unsqueeze(0)

        signal_squared_sum = audio.square().sum()
        colored_noise_squared_sum = colored_noise.square().sum()

        # for testing noise signals only
        # otherwise the signal strength is zero, so the noise signal also gets scaled down to 0
        if signal_squared_sum == 0:
            return colored_noise
        return audio + torch.sqrt(10 ** (-snr / 10) * signal_squared_sum / colored_noise_squared_sum) * colored_noise


class AudioBirdSongPairs(Dataset):
    def __init__(self,
                 directories: List[str],
                 split: str,
                 num_batches: int = None,
                 batch_size: int = None,
                 nfft: int = 1024,
                 n_mels: int = 128,
                 fmin: float = 300.0,
                 fmax: float = 18000.0,
                 dst_sr: int = 44100,
                 blacklist: List[str] = None,
                 species_list: List[str] = None,
                 audio_transforms: torch.nn.Module = None,
                 stft_transforms: torch.nn.Module = None,
                 mel_transforms: torch.nn.Module = torch.nn.Sequential(
                     torchvision.transforms.Resize((128, 384),
                                                   interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                 ),
                 min_k_pair: int = None,
                 min_sns=None
                 ):
        super(AudioBirdSongPairs, self).__init__()

        if isinstance(directories, str):
            directories = [directories]

        if species_list is None:
            # read in bird spec paths and
            species_union = set()
            for directory in directories:
                species_union = species_union.union(os.listdir(directory))

            species = list(sorted(species_union.difference(blacklist)))

            self.species = species
        else:
            self.species = species_list
        self.min_k_pair = min_k_pair

        self.audio_transforms = audio_transforms
        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft, power=2, hop_length=nfft // 4)

        self.stft_transforms = stft_transforms
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.MelScale(
                n_mels=n_mels, sample_rate=dst_sr, n_stft=nfft // 2 + 1, f_min=fmin, f_max=fmax
            ),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )
        self.mel_transforms = mel_transforms

        self.dst_sr = dst_sr

        self.number_of_elements = 0
        self.num_batches = num_batches
        self.batch_size = batch_size

        self.grouped_examples = defaultdict(list)
        for directory in directories:
            for sp in self.species:
                if sp in blacklist:
                    continue

                # get all files corresponding to train or val
                if not os.path.exists(os.path.join(directory, sp, split)):
                    continue
                files = os.listdir(os.path.join(directory, sp, split))

                if min_sns is not None:
                    def _has_min_sns(file: str):
                        sns = int(file.split("_")[1])
                        return sns > min_sns

                    files = list(filter(_has_min_sns, files))

                self.number_of_elements += len(files)

                self.grouped_examples[sp].extend(
                    [os.path.join(directory, sp, split, file)
                     for file in files
                     if os.path.splitext(file)[1] == ".wav"]
                )

        assert (self.number_of_elements == sum(
            [len(files) for files in self.grouped_examples.values()]
        ))

    def audio_to_spec(self, audio, sr):
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.dst_sr)
        if self.audio_transforms:
            audio = self.audio_transforms(audio)

        spec = self.stft(audio)
        if self.stft_transforms:
            spec = self.stft_transforms(spec)

        mel = self.mel(spec)

        if self.mel_transforms:
            mel = self.mel_transforms(mel)

        return mel

    def __len__(self):
        if self.num_batches and self.batch_size:
            return self.num_batches * self.batch_size
        if self.min_k_pair is not None:
            return self.min_k_pair * len(self.species) ** 2
        return self.number_of_elements

    def __getitem__(self, index):
        """
                    For every example, we will select two images. There are two cases,
                    positive and negative examples. For positive examples, we will have two
                    images from the same class. For negative examples, we will have two images
                    from different classes.
                    Given an index, if the index is even, we will pick the second image from the same class,
                    but it won't be the same image we chose for the first class. This is used to ensure the positive
                    example isn't trivial as the network would easily distinguish the similarity between same images. However,
                    if the network were given two different images from the same class, the network will need to learn
                    the similarity between two different images representing the same class. If the index is odd, we will
                    pick the second image from a different class than the first image.
                """

        # pick some random class for the first image
        selected_class = random.choice(self.species)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

        # pick the index to get the first image
        path_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image
        # load image by means of PIL for example

        # image_1 = self.data[index_1].clone().float()
        audio_1, sr1 = torchaudio.load(path_1)
        spec_1 = self.audio_to_spec(audio_1, sr1)

        # spec_1 = self.spec(audio_1)
        # spec_1 = self.mel_scale(spec_1)
        # spec_1 = self.spec_aug(spec_1)
        # spec_1 = self.power_to_db(spec_1)

        # image_1 = torchvision.io.read_image(path_1, mode=torchvision.io.ImageReadMode.GRAY).clone().float()
        #
        # # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

            # ensure that the index of the second image isn't the same as the first image
            if len(self.grouped_examples[selected_class]) != 1:
                while random_index_2 == random_index_1:
                    random_index_2 = random.randint(0, len(self.grouped_examples[selected_class]) - 1)

            # pick the index to get the second image
            path_2 = self.grouped_examples[selected_class][random_index_2]

            # get the second image
            audio_2, sr2 = torchaudio.load(path_2)
            spec_2 = self.audio_to_spec(audio_2, sr2)

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)

        # different class
        else:
            # pick a random class
            other_selected_class = random.choice(self.species)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.choice(self.species)

            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, len(self.grouped_examples[other_selected_class]) - 1)

            # pick the index to get the second image
            path_2 = self.grouped_examples[other_selected_class][random_index_2]

            # get the second image
            audio_2, sr2 = torchaudio.load(path_2)
            spec_2 = self.audio_to_spec(audio_2, sr2)

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        return spec_1, spec_2, target



class AudioRandomBirdDataset(Dataset):
    def __init__(self,
                 data_dirs: List[str],
                 split: str,
                 num_batches: int = None,
                 batch_size: int = None,
                 nfft: int = 1024,
                 n_mels: int = 128,
                 fmin: float = 300.0,
                 fmax: float = 18000.0,
                 dst_sr: int = 44100,
                 min_sns: float = None,
                 audio_transforms: torch.nn.Module = None,
                 stft_transforms: torch.nn.Module = None,
                 species_list: List[str] = None,
                 mel_transforms: torch.nn.Module = torch.nn.Sequential(
                     torchvision.transforms.Resize((128, 384),
                                                   interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                 ),
                 target_encoding: TargetEncoding = TargetEncoding.ONE_HOT
                 ):
        super(AudioRandomBirdDataset, self).__init__()


        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        # read in bird spec paths and
        if species_list is None:
            species_union = set()
            for data_dir in data_dirs:
                species_union = species_union.union([species for species in os.listdir(data_dir)])

            species = list(sorted(species_union))

            self.species = species
        else:
            self.species = species_list

        self.audio_transforms = audio_transforms
        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft, power=2, hop_length=nfft // 4)

        self.stft_transforms = stft_transforms
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.MelScale(
                n_mels=n_mels, sample_rate=dst_sr, n_stft=nfft // 2 + 1, f_min=fmin, f_max=fmax
            ),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )
        self.mel_transforms = mel_transforms

        self.dst_sr = dst_sr

        self.number_of_elements = 0
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.target_encoding = target_encoding

        self.grouped_examples = defaultdict(list)

        for data_dir in data_dirs:
            for sp in self.species:

                # get all files corresponding to train or val
                if not os.path.exists(os.path.join(data_dir, sp, split)):
                    continue

                # get all files corresponding to train or val
                files = os.listdir(os.path.join(data_dir, sp, split))

                if min_sns is not None:
                    def _has_min_sns(file: str):
                        sns = int(file.split("_")[1])
                        return sns > min_sns

                    files = list(filter(_has_min_sns, files))

                self.number_of_elements += len(files)

                self.grouped_examples[sp].extend(
                    [
                        os.path.join(data_dir, sp, split, file)
                        for file in files
                        if os.path.isfile(os.path.join(data_dir, sp, split, file))
                    ])
        assert(self.number_of_elements == sum([len(paths) for paths in self.grouped_examples.values()]))
        assert(all([len(paths) != 0 for paths in self.grouped_examples.values()]))

    def audio_to_spec(self, audio, sr):
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.dst_sr)
        if self.audio_transforms:
            audio = self.audio_transforms(audio)

        spec = self.stft(audio)
        if self.stft_transforms:
            spec = self.stft_transforms(spec)

        mel = self.mel(spec)
        if self.mel_transforms:
            mel = self.mel_transforms(mel)

        # normalize to [0, 1]
        # if mel.max() != mel.min():
        #     mel = (mel - mel.min()) / (mel.max() - mel.min())

        return mel

    def __len__(self):
        if self.batch_size is not None and self.iterations is not None:
            return self.batch_size * self.iterations

        return self.number_of_elements

    def __getitem__(self, index):
        """
            For every example, we will select two images. There are two cases,
            positive and negative examples. For positive examples, we will have two
            images from the same class. For negative examples, we will have two images
            from different classes.
            Given an index, if the index is even, we will pick the second image from the same class,
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn
            the similarity between two different images representing the same class. If the index is odd, we will
            pick the second image from a different class than the first image.
        """

        species_index = random.randint(0, len(self.species) - 1)
        species = self.species[species_index]

        random_index = random.randint(0, len(self.grouped_examples[species]) - 1)
        path = self.grouped_examples[species][random_index]

        audio, sr1 = torchaudio.load(path)
        spec = self.audio_to_spec(audio, sr1)

        if self.target_encoding == TargetEncoding.ONE_HOT:
            target = torch.where(torch.arange(0, len(self.species)).eq(species_index), 1.0, 0.0)
        elif self.target_encoding == TargetEncoding.CATEGORICAL:
            target = self.species[species_index]

        return spec, target



class AudioDeterministicBirdDataset(Dataset):
    def __init__(self,
                 data_dirs,
                 split: str,
                 num_batches: int = None,
                 batch_size: int = None,
                 nfft: int = 1024,
                 n_mels: int = 128,
                 fmin: float = 300.0,
                 fmax: float = 18000.0,
                 dst_sr: int = 44100,
                 min_sns: float = None,
                 species_list: List[str] = None,
                 audio_transforms: torch.nn.Module = None,
                 stft_transforms: torch.nn.Module = None,
                 mel_transforms: torch.nn.Module = torch.nn.Sequential(
                     torchvision.transforms.Resize((128, 384),
                                                   interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                 ),
                 target_encoding: TargetEncoding = TargetEncoding.ONE_HOT
                 ):
        super(AudioDeterministicBirdDataset, self).__init__()

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        if species_list is None:
            # read in bird spec paths and
            species_union = set()
            for directory in data_dirs:
                species_union = species_union.union([species for species in os.listdir(directory)])

            species = list(sorted(species_union))

            self.species = species
        else:
            self.species = species_list
        self.grouped_examples = {}
        self.audio_transforms = audio_transforms
        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft, power=2, hop_length=nfft // 4)

        self.stft_transforms = stft_transforms
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.MelScale(
                n_mels=n_mels, sample_rate=dst_sr, n_stft=nfft // 2 + 1, f_min=fmin, f_max=fmax
            ),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )
        self.mel_transforms = mel_transforms

        self.dst_sr = dst_sr

        self.number_of_elements = 0
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.target_encoding = target_encoding

        self.idx_to_path_sp_index = {}

        current_idx = 0

        for directory in data_dirs:
            for species in self.species:
                sp_index = self.species.index(species)

                # get all files corresponding to train or val
                if not os.path.exists(os.path.join(directory, species, split)):
                    continue

                # get all files corresponding to train or val
                files = os.listdir(os.path.join(directory, species, split))

                if min_sns is not None:
                    def _has_min_sns(file: str):
                        sns = int(file.split("_")[1])
                        return sns > min_sns

                    files = list(filter(_has_min_sns, files))

                self.number_of_elements += len(files)

                for file in files:
                    path = os.path.join(directory, species, split, file)
                    self.idx_to_path_sp_index[current_idx] = (path, sp_index)
                    current_idx += 1
        assert (self.number_of_elements == len(self.idx_to_path_sp_index))

    def audio_to_spec(self, audio, sr):
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.dst_sr)
        if self.audio_transforms:
            audio = self.audio_transforms(audio)

        spec = self.stft(audio)
        if self.stft_transforms:
            spec = self.stft_transforms(spec)

        mel = self.mel(spec)
        if self.mel_transforms:
            mel = self.mel_transforms(mel)

        return mel

    def __len__(self):
        return self.number_of_elements

    def __getitem__(self, index):
        """
            For every example, we will select two images. There are two cases,
            positive and negative examples. For positive examples, we will have two
            images from the same class. For negative examples, we will have two images
            from different classes.
            Given an index, if the index is even, we will pick the second image from the same class,
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn
            the similarity between two different images representing the same class. If the index is odd, we will
            pick the second image from a different class than the first image.
        """

        path, species_index = self.idx_to_path_sp_index[index]

        audio, sr1 = torchaudio.load(path)
        spec = self.audio_to_spec(audio, sr1)

        if self.target_encoding == TargetEncoding.ONE_HOT:
            target = torch.where(torch.arange(0, len(self.species)).eq(species_index), 1.0, 0.0)
        elif self.target_encoding == TargetEncoding.CATEGORICAL:
            target = self.species[species_index]

        return spec, target



class AudioCappedDeterministicBirdDataset(Dataset):
    def __init__(self,
                 data_dirs,
                 split: str,
                 max_num_per_class: int = None,
                 num_batches: int = None,
                 batch_size: int = None,
                 nfft: int = 1024,
                 n_mels: int = 128,
                 fmin: float = 300.0,
                 fmax: float = 18000.0,
                 dst_sr: int = 44100,
                 min_sns: float = None,
                 species_list: List[str] = None,
                 audio_transforms: torch.nn.Module = None,
                 stft_transforms: torch.nn.Module = None,
                 mel_transforms: torch.nn.Module = torch.nn.Sequential(
                     torchvision.transforms.Resize((128, 384),
                                                   interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                 ),
                 target_encoding: TargetEncoding = TargetEncoding.ONE_HOT
                 ):
        super(AudioCappedDeterministicBirdDataset, self).__init__()


        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        if species_list is None:
            # read in bird spec paths and
            species_union = set()
            for directory in data_dirs:
                species_union = species_union.union([species for species in os.listdir(directory)])

            species = list(sorted(species_union))

            self.species = species
        else:
            self.species = species_list
        self.grouped_examples = {}
        self.audio_transforms = audio_transforms
        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft, power=2, hop_length=nfft // 4)

        self.stft_transforms = stft_transforms
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.MelScale(
                n_mels=n_mels, sample_rate=dst_sr, n_stft=nfft // 2 + 1, f_min=fmin, f_max=fmax
            ),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )
        self.mel_transforms = mel_transforms

        self.dst_sr = dst_sr

        self.number_of_elements = 0
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.target_encoding = target_encoding

        self.idx_to_path_sp_index = {}

        current_idx = 0

        num_samples_per_class = defaultdict(int)

        for directory in data_dirs:
            for species in self.species:
                sp_index = self.species.index(species)

                # get all files corresponding to train or val
                if not os.path.exists(os.path.join(directory, species, split)):
                    continue

                # get all files corresponding to train or val
                files = os.listdir(os.path.join(directory, species, split))

                if min_sns is not None:
                    def _has_min_sns(file: str):
                        sns = int(file.split("_")[1])
                        return sns > min_sns

                    files = list(filter(_has_min_sns, files))

                if num_samples_per_class is not None:
                    remaining_samples = max_num_per_class - num_samples_per_class[sp_index]
                    if len(files) > remaining_samples:
                        files = random.sample(files, remaining_samples)
                    num_samples_per_class[sp_index] += len(files)

                self.number_of_elements += len(files)

                for file in files:
                    path = os.path.join(directory, species, split, file)
                    self.idx_to_path_sp_index[current_idx] = (path, sp_index)
                    current_idx += 1
        assert (self.number_of_elements == len(self.idx_to_path_sp_index))

    def audio_to_spec(self, audio, sr):
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.dst_sr)
        if self.audio_transforms:
            audio = self.audio_transforms(audio)

        spec = self.stft(audio)
        if self.stft_transforms:
            spec = self.stft_transforms(spec)

        mel = self.mel(spec)
        if self.mel_transforms:
            mel = self.mel_transforms(mel)

        # normalize to [0, 1]
        # if mel.max() != mel.min():
        #     mel = (mel - mel.min()) / (mel.max() - mel.min())

        return mel

    def __len__(self):
        return self.number_of_elements

    def __getitem__(self, index):
        """
            For every example, we will select two images. There are two cases,
            positive and negative examples. For positive examples, we will have two
            images from the same class. For negative examples, we will have two images
            from different classes.
            Given an index, if the index is even, we will pick the second image from the same class,
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn
            the similarity between two different images representing the same class. If the index is odd, we will
            pick the second image from a different class than the first image.
        """

        path, species_index = self.idx_to_path_sp_index[index]

        audio, sr1 = torchaudio.load(path)
        spec = self.audio_to_spec(audio, sr1)

        if self.target_encoding == TargetEncoding.ONE_HOT:
            target = torch.where(torch.arange(0, len(self.species)).eq(species_index), 1.0, 0.0)
        elif self.target_encoding == TargetEncoding.CATEGORICAL:
            target = self.species[species_index]

        return spec, target


class AudioDeterministicESCDataset(Dataset):
    def __init__(self,
                 directory: str,
                 dst_sr: int,
                 classes: List[str] = None,
                 num_batches: int = None,
                 batch_size: int = None,
                 nfft: int = 1024,
                 n_mels: int = 128,
                 fmin: float = 300.0,
                 fmax: float = 18000.0,
                 audio_transforms: torch.nn.Module = None,
                 stft_transforms: torch.nn.Module = None,
                 mel_transforms: torch.nn.Module = torch.nn.Sequential(
                     torchvision.transforms.Resize((128, 384),
                                                   interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                 ),
                 target_encoding: TargetEncoding = TargetEncoding.ONE_HOT
                 ):
        super(AudioDeterministicESCDataset, self).__init__()

        self.classes = classes

        meta_path = os.path.join(directory, "meta", "esc50.csv")
        audio_dir = os.path.join(directory, "audio")

        self.grouped_examples = defaultdict(lambda: [])

        self.audio_transforms = audio_transforms
        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft, power=2, hop_length=nfft // 4)

        self.stft_transforms = stft_transforms
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.MelScale(
                n_mels=n_mels, sample_rate=dst_sr, n_stft=nfft // 2 + 1, f_min=fmin, f_max=fmax
            ),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )
        self.mel_transforms = mel_transforms

        self.dst_sr = dst_sr

        self.number_of_elements = 0
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.target_encoding = target_encoding

        self.idx_to_path_class_index = {}

        current_idx = 0
        with open(meta_path, "r") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                category = row["category"]
                if category in classes:
                    index = classes.index(category)
                    path = os.path.join(audio_dir, row["filename"])
                    self.idx_to_path_class_index[current_idx] = (path, index)

                    current_idx += 1
        assert (current_idx == len(self.idx_to_path_class_index))

    def audio_to_spec(self, audio, sr):
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.dst_sr)
        if self.audio_transforms:
            audio = self.audio_transforms(audio)

        spec = self.stft(audio)
        if self.stft_transforms:
            spec = self.stft_transforms(spec)

        mel = self.mel(spec)
        if self.mel_transforms:
            mel = self.mel_transforms(mel)

        return mel

    def __len__(self):
        return len(self.idx_to_path_class_index)

    def __getitem__(self, index):
        """
            For every example, we will select two images. There are two cases,
            positive and negative examples. For positive examples, we will have two
            images from the same class. For negative examples, we will have two images
            from different classes.
            Given an index, if the index is even, we will pick the second image from the same class,
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn
            the similarity between two different images representing the same class. If the index is odd, we will
            pick the second image from a different class than the first image.
        """

        path, species_index = self.idx_to_path_class_index[index]

        audio, sr1 = torchaudio.load(path)
        # take first two seconds
        audio = audio[:, :2*sr1]

        spec = self.audio_to_spec(audio, sr1)

        if self.target_encoding == TargetEncoding.ONE_HOT:
            target = torch.where(torch.arange(0, len(self.classes)).eq(species_index), 1.0, 0.0)
        elif self.target_encoding == TargetEncoding.CATEGORICAL:
            target = self.classes[species_index]

        return spec, target


def read_species_list(species_list: str):
    species = []

    with open(species_list, "r") as f:
        for row in f:
            species.append(row.strip())

    return species