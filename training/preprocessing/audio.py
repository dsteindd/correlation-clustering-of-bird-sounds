"""MIT License

Copyright (c) 2018 Stefan Kahl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
# Code from: https://github.com/kahst/BirdCLEF-Baseline

import os
from multiprocessing import Pool

import cv2
import librosa
import numpy as np
import soundfile as sf

from config import SPEC_SIGNAL_THRESHOLD


def open_audio_file(path, sample_rate=44100, as_mono=True, mean_substract=False):
    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, mono=as_mono)

    # Noise reduction?
    if mean_substract:
        sig -= sig.mean()

    return sig, rate


def split_signal(sig, rate, seconds, overlap, minlen):
    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break

        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, np.zeros((int(rate * seconds) - len(split),))))

        sig_splits.append(split)

    return sig_splits


def melspec(sig, rate, shape=(128, 256), fmin=500, fmax=15000, normalize=True, preemphasis=0.95):
    # shape = (height, width) in pixels

    # Mel-Spec parameters
    SAMPLE_RATE = rate
    N_FFT = shape[0] * 8  # = window length
    N_MELS = shape[0]
    HOP_LEN = len(sig) // (shape[1] - 1)
    FMAX = fmax
    FMIN = fmin

    # Preemphasis as in python_speech_features by James Lyons
    if preemphasis:
        sig = np.append(sig[0], sig[1:] - preemphasis * sig[:-1])

    # Librosa mel-spectrum
    melspec = librosa.feature.melspectrogram(y=sig, sr=SAMPLE_RATE, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS,
                                             fmax=FMAX, fmin=FMIN, power=1.0)

    # Convert power spec to dB scale (compute dB relative to peak power)
    melspec = librosa.amplitude_to_db(melspec, ref=np.max, top_db=80)

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    melspec = melspec[::-1, ...]

    # Trim to desired shape if too large
    melspec = melspec[:shape[0], :shape[1]]

    # Normalize values between 0 and 1
    if normalize:
        melspec -= melspec.min()
        if not melspec.max() == 0:
            melspec /= melspec.max()
        else:
            mlspec = np.clip(melspec, 0, 1)

    return melspec.astype('float32')


def stft(sig, rate, shape=(128, 256), fmin=500, fmax=15000, normalize=True):
    # shape = (height, width) in pixels

    # STFT-Spec parameters
    N_FFT = int((rate * shape[0] * 2) / abs(fmax - fmin)) + 1
    P_MIN = int(float(N_FFT / 2) / rate * fmin) + 1
    P_MAX = int(float(N_FFT / 2) / rate * fmax) + 1
    HOP_LEN = len(sig) // (shape[1] - 1)

    # Librosa stft-spectrum
    spec = librosa.core.stft(sig, hop_length=HOP_LEN, n_fft=N_FFT, window='hamm')

    # Convert power spec to dB scale (compute dB relative to peak power)
    spec = librosa.amplitude_to_db(librosa.core.magphase(spec)[0], ref=np.max, top_db=80)

    # Trim to desired shape using cutoff frequencies
    spec = spec[P_MIN:P_MAX, :shape[1]]

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    spec = spec[::-1, ...]

    # Normalize values between 0 and 1
    if normalize:
        spec -= spec.min()
        if not spec.max() == 0:
            spec /= spec.max()
        else:
            spec = np.clip(spec, 0, 1)

    return spec.astype('float32')


def get_spec(sig, rate, shape, spec_type='linear', **kwargs):
    if spec_type.lower() == 'melspec':
        return melspec(sig, rate, shape, **kwargs)
    else:
        return stft(sig, rate, shape, **kwargs)


def signal2noise(spec):
    # Get working copy
    spec = spec.copy()

    # spec = mixture_denoise(spec, threshold=1.5)

    # Calculate median for columns and rows
    col_median = np.median(spec, axis=0, keepdims=True)
    row_median = np.median(spec, axis=1, keepdims=True)

    # Binary threshold
    spec[spec < row_median * 1.25] = 0.0
    spec[spec < col_median * 1.15] = 0.0
    spec[spec > 0] = 1.0

    # Median blur
    spec = cv2.medianBlur(spec, 3)

    # Morphology
    spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE, np.ones((3, 3), np.float32))

    # Sum of all values
    spec_sum = spec.sum()

    # Signal to noise ratio (higher is better)
    try:
        s2n = spec_sum / (spec.shape[0] * spec.shape[1] * spec.shape[2])
    except:
        s2n = spec_sum / (spec.shape[0] * spec.shape[1])

    return s2n


def specs_from_signal(sig, rate, shape, seconds, overlap, minlen, **kwargs):
    # Split signal in consecutive chunks with overlap
    sig_splits = split_signal(sig, rate, seconds, overlap, minlen)

    # Extract specs for every sig split
    for sig in sig_splits:
        # Get spec for signal chunk
        spec = get_spec(sig, rate, shape, **kwargs)

        yield sig, spec


def specsFromFile(path, rate, seconds, overlap, minlen, shape, start=-1, end=-1, **kwargs):
    # Open file
    sig, rate = open_audio_file(path, rate)

    # Trim signal?
    if start > -1 and end > -1:
        sig = sig[int(start * rate):int(end * rate)]
        minlen = 0

    # Yield all specs for file
    for (sig, spec) in specs_from_signal(sig, rate, shape, seconds, overlap, minlen, **kwargs):
        yield (sig, spec)


def _get_specs_for_species(species: str, split: str, directory: str, out_dir: str):

    spec_out_dir = os.path.join(out_dir, "specs")
    audio_out_dir = os.path.join(out_dir, "audio")

    # print(f"Species/Split '{species}' / '{split}'...")
    species_dir = os.path.join(directory, species, split)
    for file in os.listdir(species_dir):
        count = 0

        file_path = os.path.join(species_dir, file)
        file_id = os.path.splitext(file)[0]

        for (sig, spec) in specsFromFile(
                file_path,
                rate=44100,
                seconds=2,
                overlap=0.0,
                minlen=2,
                shape=(128, 384),
                fmin=300,
                fmax=18000,
                spec_type='melspec'
        ):
            noise = signal2noise(spec)

            if noise > SPEC_SIGNAL_THRESHOLD:
                spec_path = os.path.join(spec_out_dir, species, split)
                if not os.path.exists(spec_path):
                    os.makedirs(spec_path)

                noise = int(noise*1000)

                cv2.imwrite(os.path.join(spec_path, f"{file_id}_{noise}_{count}.png"), spec * 255.0)

                audio_path = os.path.join(audio_out_dir, species, split)
                if not os.path.exists(audio_path):
                    os.makedirs(audio_path)

                audio_fn_path = os.path.join(audio_path, f"{file_id}_{noise}_{count}.wav")
                sf.write(audio_fn_path, sig, 44100)

                count += 1


def parse_directory(
        directory: str,
        out_dir: str
):
    splits = ["train", "val", "test"]

    # spec_out_dir = os.path.join(out_dir, "specs")
    # audio_out_dir = os.path.join(out_dir, "audio")

    # assemble list of jobs
    subdir_jobs = []
    for split in splits:
        for species in os.listdir(directory):
            subdir_jobs.append((species, split, directory, out_dir))

    with Pool(os.cpu_count()) as p:
        p.starmap(_get_specs_for_species, subdir_jobs)


def get_stats(directory: str):

    stats = {}
    for species in os.listdir(directory):
        stats[species] = {}
        species_dir = os.path.join(directory, species)
        for split in os.listdir(species_dir):
            stats[species][split] = len(os.listdir(os.path.join(species_dir, split)))

    for (species, st) in stats.items():
        for (split, num) in st.items():
            print(f"{species} / {split}: ", num)


if __name__ == '__main__':
    parse_directory(
        directory="../data/split-3600-A/",
        out_dir="../data/split-3600-A-chunks/"
    )
