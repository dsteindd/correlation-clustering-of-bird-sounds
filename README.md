# Correlation Clustering of Bird Sounds

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23644014.svg)](https://doi.org/10.5281/zenodo.23644014)

This repository contains the code used for the paper
[Correlation Clustering of Bird Sounds](https://doi.org/10.1007/978-3-031-54605-1_33)
(GCPR 2023, preprint: [arXiv:2306.09906](https://arxiv.org/abs/2306.09906)).

The data and trained models used in the paper are not part of this repository.

## Contents

- `training/`: Python/PyTorch code for downloading recordings from
  [xeno-canto](https://xeno-canto.org), preprocessing them into spectrograms, and training and
  analysing siamese and classification networks.
  - `preprocessing/download.py`: download of recordings and metadata from xeno-canto
  - `preprocessing/audio.py`, `preprocessing/data_split.py`: spectrogram extraction and data splits
  - `torch_siamese.py`, `torch_classification.py`: training
  - `torch_siamese_analysis.py`, `torch_classifier_analysis.py`: evaluation
- `correlation-clustering/`: C++ code for correlation clustering of the network outputs
  (greedy additive edge contraction and Kernighan-Lin) and its evaluation.

## Building the C++ code

Requires CMake >= 3.10 and a C++17 compiler.

```bash
cmake -S correlation-clustering -B build
cmake --build build
```

## Downloading recordings

The xeno-canto API (v3) requires a personal API key, which is shown on your
[xeno-canto account page](https://xeno-canto.org/account). Pass it as an environment variable
and never commit it. As configured in its `__main__` block, `download.py` downloads all
recordings from Germany:

```bash
export XENO_CANTO_API_KEY=<your key>
cd training/preprocessing && python download.py
```

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{stein-2023-correlation,
    author = {David Stein and Bjoern Andres},
    title = {Correlation Clustering of Bird Sounds},
    year = {2023},
    booktitle = {German Conference on Pattern Recognition (GCPR)},
    doi = {10.1007/978-3-031-54605-1_33},
}
```

Metadata for citing the software itself is in [`CITATION.cff`](CITATION.cff).

## License
The code is licensed under the MIT License, see [LICENSE](LICENSE).
Most headers in ```correlation-clustering/include/andres``` are taken from the
[graph library of Bjoern Andres](http://www.andres.sc/graph.html) and are distributed under its license;
[correlation-clustering/include/andres/LICENSE](correlation-clustering/include/andres/LICENSE) lists them.
Parts of ```training/preprocessing/audio.py``` and ```training/preprocessing/config.py``` are taken from
[BirdCLEF-Baseline](https://github.com/kahst/BirdCLEF-Baseline) by Stefan Kahl (MIT License).
