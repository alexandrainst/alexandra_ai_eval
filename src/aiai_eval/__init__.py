"""
.. include:: ../../README.md
"""

import logging
import os

import colorama
import pkg_resources
from termcolor import colored

from .evaluator import Evaluator  # noqa
from .utils import block_terminal_output

# Fetches the version of the package as defined in pyproject.toml
__version__ = pkg_resources.get_distribution("aiai_eval").version


# Block unwanted terminal outputs
block_terminal_output()


# Ensure that termcolor also works on Windows
colorama.init()


# Set up logging
fmt = colored("%(asctime)s [%(levelname)s] <%(name)s>\n↳ ", "cyan") + colored(
    "%(message)s", "yellow"
)
logging.basicConfig(level=logging.INFO, format=fmt)


# Disable parallelisation when tokenizing, as that can lead to errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Enable MPS fallback to CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Tell Windows machines to use UTF-8 encoding
os.environ["ConEmuDefaultCp"] = "65001"
os.environ["PYTHONIOENCODING"] = "UTF-8"
