"""Enums used in the project."""

import enum


class Device(str, enum.Enum):
    """The compute device to use for the evaluation.

    Attributes:
        CPU:
            CPU device.
        MPS:
            MPS GPU, used in M-series MacBooks.
        CUDA:
            CUDA GPU, used with NVIDIA GPUs.
    """

    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


class Framework(str, enum.Enum):
    """The framework of a model.

    Attributes:
        PYTORCH:
            PyTorch framework.
        JAX:
            JAX framework.
        SPACY:
            spaCy framework.
    """

    PYTORCH = "pytorch"
    JAX = "jax"
    SPACY = "spacy"
