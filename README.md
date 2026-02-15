<!-- ![Icon](icon1.jpeg) -->
<div align="center">
<img src="icon.jpeg" width="200" height="200" alt="Alt text" title="Optional title">
</div>


# Introduction

This is an End-to-End (E2E) Automatic Speech Recognition (ASR) toolkit based on Differentiable Weighted Finite-State Transducer (DWFST) with flexible topology definition support, named **BenNevis**, which is essentially implemented with [PyTorch](https://github.com/pytorch/pytorch) and [k2](https://github.com/k2-fsa/k2), where the former serves as the Neural Network backend and the latter for DWFST.

**Note that this project is still under development, so code may be changed rapidly in the near future**

# Features

* Flexible **topology** support
* Kaldi-style data manipulation with [kaldiio](https://github.com/nttcslab-sp/kaldiio) and kaldi scripts
* DWFST backed by [k2](https://github.com/k2-fsa/k2)
* Pure [PyTorch](https://github.com/pytorch/pytorch) implementation
* Distributed Data Parallel (DDP) support with `torch.distributed`
* [WandB](https://wandb.ai/) logger support

# Installation

## Prerequisites

- Python >= 3.8
- CUDA-capable GPU (recommended for training)
- [Kaldi](https://github.com/kaldi-asr/kaldi) (for data preparation and decoding)

## Quick Install (Recommended)

### 0. Clone the Repository

```shell
git clone https://github.com/zeyuzhao/BenNevis.git
cd BenNevis
```

### 1. Install Kaldi

Install [Kaldi](https://github.com/kaldi-asr/kaldi) and link it to BenNevis:
```shell
cd tools
./put_kaldi.sh /path/to/kaldi
```
This creates a symbolic link to your Kaldi installation. Kaldi is used for data preparation, feature extraction, graph compilation (OpenFST), and WFST-based decoding.

### 2. Install BenNevis

```shell
# Create virtual environment
python3 -m venv tools/venv # Create a virtual environment in tools/venv
source tools/venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust CUDA and PyTorch version as needed)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install k2 (must match PyTorch and CUDA versions)
pip install k2==1.24.4.dev20231220+cuda11.8.torch2.1.0 -f https://k2-fsa.github.io/k2/cuda.html

# Install BenNevis
pip install -e .

# For development
pip install -e ".[dev]"
```

See [k2 installation page](https://k2-fsa.github.io/k2/cuda.html) for available precompiled wheels matching your PyTorch and CUDA versions.

## Alternative: Automated Setup Script

For quick setup with default versions:
```shell
cd tools
./create_env.sh
```

This creates a virtual environment with PyTorch 2.1.0 and k2. You can modify the script to use different versions.

## Installation from Requirements

If you prefer using requirements files:

```shell
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

**Note:** You still need to install k2 separately with the correct PyTorch/CUDA version match.

## Verify Installation

```python
import BenNevis
print(BenNevis.__version__)  # Should print: 0.1.0

# Test imports
from BenNevis import Trainer, Dataset, Lang, GraphLoss
```

# Quick Start

The best way to start is to look at the YesNo recipe in `egs/yesno/asr/run.sh`, where you can see the basic workflow in BenNevis.

It usually includes

1. Data preparation
2. Feature extraction
3. Training graph preparation (for various topologies)
4. Decoding graph preparation (for various topologies)
5. Training (based on PyTorch Distributed Data Parallel, see `run/train.py` for details)
6. Predicting (obtaining the posterior from the NN model, see `run/predict.sh` for details)
7. Decoding (taking the decoding graph and the posterior as inputs, see `run/decode_faster.sh` for details)
8. Alignment (optional, supporting alignment with ground truth or decoding results, see `run/align.sh` for details)

# Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black BenNevis/
isort BenNevis/
```

# Citation

If you use BenNevis in your research, please cite it:

```bibtex
@software{bennevis2024,
  author = {Zhao, Zeyu},
  title = {BenNevis: End-to-End ASR Toolkit based on DWFST},
  year = {2024},
  url = {https://github.com/zeyuzhao/BenNevis},
  version = {0.1.0}
}
```

You can also use the [CITATION.cff](CITATION.cff) file for automatic citation generation.

# Contact

Please feel free to contact me by [email](mailto:zeyuhongwu1995@gmail.com) for any issue about BenNevis.
