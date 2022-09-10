"""Enums used in the project."""

import enum

from .country_codes import ALL_COUNTRY_CODES


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


country_code_enum_list = [("EMPTY", "")] + [
    (country_code, country_code.lower()) for country_code in ALL_COUNTRY_CODES
]
CountryCode = enum.Enum("CountryCode", country_code_enum_list)  # type: ignore[misc]
