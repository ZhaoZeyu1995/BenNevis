"""
BenNevis: End-to-End ASR Toolkit based on DWFST

BenNevis is an End-to-End Automatic Speech Recognition (ASR) toolkit
based on Differentiable Weighted Finite-State Transducer (DWFST) with
flexible topology definition support.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

from typing import TYPE_CHECKING

__version__ = "0.1.0"
__author__ = "Zeyu Zhao"
__email__ = "zeyuhongwu1995@gmail.com"
__license__ = "Apache-2.0"

if TYPE_CHECKING:
    from BenNevis.core.dataset import CollateFunc, Dataset
    from BenNevis.core.lang import Lang
    from BenNevis.core.losses import GraphLoss
    from BenNevis.core.trainer import Trainer


def __getattr__(name):
    if name in ("Dataset", "CollateFunc"):
        from BenNevis.core.dataset import CollateFunc, Dataset

        return {"Dataset": Dataset, "CollateFunc": CollateFunc}[name]
    if name == "Lang":
        from BenNevis.core.lang import Lang

        return Lang
    if name == "GraphLoss":
        from BenNevis.core.losses import GraphLoss

        return GraphLoss
    if name == "Trainer":
        from BenNevis.core.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Trainer",
    "Dataset",
    "CollateFunc",
    "Lang",
    "GraphLoss",
    "__version__",
]
