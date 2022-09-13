"""Utility functions for the project."""

import gc
import importlib
import logging
import os
import random
import re
import warnings
from typing import List, Optional, Sequence, Union

import numpy as np
import pkg_resources
import requests
import torch
from datasets.utils import disable_progress_bar
from requests import RequestException
from wasabi import msg as wasabi_msg

from .enums import Device, Framework
from .exceptions import InvalidArchitectureForTask

logger = logging.getLogger(__name__)


def has_integers(seq: Sequence) -> bool:
    """Checks if a sequence contains only integers.

    Args:
        seq (Sequence):
            The sequence to check.

    Returns:
        bool:
            Whether the sequence contains only integers.
    """
    return np.asarray(seq).dtype.kind == "i"


def has_floats(seq: Sequence) -> bool:
    """Checks if a sequence contains only floats.

    Args:
        seq (Sequence):
            The sequence to check.

    Returns:
        bool:
            Whether the sequence contains only floats.
    """
    return np.asarray(seq).dtype.kind == "f"


def clear_memory():
    """Clears the memory of unused items."""

    # Clear the Python cache
    gc.collect()

    # Empty the CUDA cache
    # TODO: Also empty MPS cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def enforce_reproducibility(
    framework: Framework, seed: int = 703
) -> np.random.Generator:
    """Ensures reproducibility of experiments.

    Args:
        framework (Framework):
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
    if framework in [Framework.PYTORCH, Framework.JAX]:
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
    logging.getLogger("codecarbon").setLevel(logging.ERROR)

    # Disable `wasabi` logging, used in spaCy
    wasabi_msg.no_print = True

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


def check_supertask(architectures: Sequence[str], supertask: str):
    """Checks if the supertask corresponds to the architectures.

    Args:
        architectures (list of str):
            The model architecture names.
        supertask (str):
            The supertask associated to a task written in kebab-case, e.g.,
            text-classification.

    Raises:
        InvalidArchitectureForTask:
            If the PascalCase version of the supertask is not found in any of the
            architectures.
    """
    # Create boolean variable that checks if the supertask exists among the available
    # architectures
    supertask_is_an_architecture = any(
        kebab_to_pascal(supertask) in architecture for architecture in architectures
    )

    # If the supertask is not an architecture, raise an error
    if not supertask_is_an_architecture:
        raise InvalidArchitectureForTask(
            architectures=architectures, supertask=supertask
        )


def get_class_by_name(
    class_name: Union[str, Sequence[str]],
    module_name: Optional[str] = None,
) -> Union[None, type]:
    """Get a class by its name.

    Args:
        class_name (str or list of str):
            The name of the class, written in kebab-case. The corresponding class name
            must be the same, but written in PascalCase, and lying in a module with the
            same name, but written in snake_case. If a list of strings is passed, the
            first class that is found is returned.
        module_name (str, optional):
            The name of the module where the class is located. If None then the module
            name is assumed to be the same as the class name, but written in
            snake_case. Defaults to None.

    Returns:
        type or None:
            The class. If the class is not found, None is returned.
    """
    # Ensure that `class_name` is a sequence
    if isinstance(class_name, str):
        class_name = [class_name]

    # Loop over the class names
    for name in class_name:

        # Get the snake_case and PascalCase version of the class name
        name_snake = name.replace("-", "_")
        name_pascal = kebab_to_pascal(name)

        # Import the module
        try:
            if not module_name:
                module_name = f"aiai_eval.{name_snake}"
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            module_name = None
            continue

        # Get the class from the module
        try:
            class_ = getattr(module, name_pascal)
        except AttributeError:
            module_name = None
            continue

        # Return the class
        return class_

    # If the class could not be found, return None
    return None


def kebab_to_pascal(kebab_string: str) -> str:
    """Converts a kebab-case string to PascalCase.

    Args:
        kebab_string (str):
            The kebab-case string.

    Returns:
        str:
            The PascalCase string.
    """
    return "".join(word.title() for word in kebab_string.split("-"))
