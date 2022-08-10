"""Utility functions for the project."""

import os
import random

import numpy as np
import torch


def enforce_reproducibility(framework: str, seed: int = 703):
    """Ensures reproducibility of experiments.

    Args:
        framework (str):
            The framework used for the benchmarking.
        seed (int):
            Seed for the random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if framework in ("pytorch", "jax"):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    return rng
