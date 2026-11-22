"""
Random seed setting

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

import random

import numpy as np
import torch


def setup_seed(seed: int = 1368):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
