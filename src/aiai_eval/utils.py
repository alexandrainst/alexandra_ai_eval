"""Utility functions for the project."""

import enum
import gc
import logging
import os
import random
import re
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pkg_resources
import requests
import torch
from datasets.utils import disable_progress_bar
from requests import RequestException

logger = logging.getLogger(__name__)


def clear_memory():
    """Clears the memory of unused items."""

    # Clear the Python cache
    gc.collect()

    # Empty the CUDA cache
    # TODO: Also empty MPS cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def enforce_reproducibility(framework: str, seed: int = 703) -> np.random.Generator:
    """Ensures reproducibility of experiments.

    Args:
        framework (str):
            The framework used for the benchmarking.
        seed (int):
            Seed for the random number generator.

    Returns:
        NumPy Generator object:
            A random number generator, with seed `seed`.
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


def is_module_installed(module: str) -> bool:
    """Check if a module is installed.

    This is used when dealing with spaCy models, as these are installed as separate
    Python packages.

    Args:
        module (str):
            The name of the module.

    Returns:
        bool:
            Whether the module is installed or not.
    """
    # Get list of all modules, including their versions
    installed_modules_with_versions = list(pkg_resources.working_set)

    # Strip the module versions from the list of modules. Also make the modules lower
    # case and replace dashes with underscores
    installed_modules = [
        re.sub("[0-9. ]", "", str(module)).lower().replace("-", "_")
        for module in installed_modules_with_versions
    ]

    # Check if the module is installed by checking if the module name is in the list
    return module.lower() in installed_modules


def block_terminal_output():
    """Blocks libraries from writing output to the terminal.

    This filters warnings from some libraries, sets the logging level to ERROR for some
    libraries and disables tokeniser progress bars when using Hugging Face tokenisers.
    """

    # Ignore miscellaneous warnings
    warnings.filterwarnings(
        "ignore",
        module="torch.nn.parallel*",
        message="Was asked to gather along dimension 0, but all input tensors were "
        "scalars; will instead unsqueeze and return a vector.",
    )
    warnings.filterwarnings("ignore", module="seqeval*")

    # Up the logging level, to disable outputs
    logging.getLogger("filelock").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    # Disable the tokeniser progress bars
    disable_progress_bar()


def internet_connection_available() -> bool:
    """Checks if internet connection is available by pinging google.com.

    Returns:
            bool:
                Whether or not internet connection is available.
    """
    try:
        requests.get("https://www.google.com")
        return True
    except RequestException:
        return False


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


def get_available_devices() -> List[Device]:
    """Gets the available devices.

    This will check whether a CUDA GPU and MPS GPU is available.

    Returns:
        list of Device objects:
            The available devices, sorted as CUDA, MPS, CPU.
    """
    available_devices = list()

    # Add CUDA to the list if it is available
    if torch.cuda.is_available():
        available_devices.append(Device.CUDA)

    # Add MPS to the list if it is available
    if torch.backends.mps.is_available():
        available_devices.append(Device.MPS)

    # Always add CPU to the list
    available_devices.append(Device.CPU)

    # Return the list of available devices
    return available_devices


@dataclass
class Label:
    """A label in a dataset task.

    Attributes:
        name (str):
            The name of the label.
        synonyms (list of str):
            The synonyms of the label.
    """

    name: str
    synonyms: List[str