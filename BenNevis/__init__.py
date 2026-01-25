"""
BenNevis: End-to-End ASR Toolkit based on DWFST

BenNevis is an End-to-End Automatic Speech Recognition (ASR) toolkit
based on Differentiable Weighted Finite-State Transducer (DWFST) with
flexible topology definition support.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

__version__ = "0.1.0"
__author__ = "Zeyu Zhao"
__email__ = "zeyuhongwu1995@gmail.com"
__license__ = "Apache-2.0"

from BenNevis.core.dataset import CollateFunc, Dataset
from BenNevis.core.lang import Lang
from BenNevis.core.losses import GraphLoss

# Expose main classes for convenience
from BenNevis.core.trainer import Trainer

__all__ = [
    "Trainer",
    "Dataset",
    "CollateFunc",
    "Lang",
    "GraphLoss",
    "__version__",
]
