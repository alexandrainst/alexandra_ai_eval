"""Utility functions for the project."""

import gc
import logging
import os
import random
import re
import warnings
from typing import Dict, Sequence, Tuple

import numpy as np
import pkg_resources
import requests
import torch
from requests import RequestException

from .config import MetricConfig

logger = logging.getLogger(__name__)


def clear_memory():
    """Clears the memory of unused items."""

    # Clear the Python cache
    gc.collect()

    # Empty the CUDA cache
    # TODO: Also empty MPS cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def log_scores(
    dataset_name: str,
    metric_configs: Sequence[MetricConfig],
    scores: Sequence[Dict[str, float]],
    model_id: str,
) -> dict:
    """Log the scores.

    Args:
        dataset_name (str):
            Name of the dataset.
        metric_configs (sequence of MetricConfig objects):
            Sequence of metrics to log.
        scores (dict):
            The scores that are to be logged.
        model_id (str):
            The full Hugging Face Hub path to the pretrained transformer model.

    Returns:
        dict:
            A dictionary with keys 'raw_scores' and 'total', with 'raw_scores' being
            identical to `scores` and 'total' being a dictionary with the aggregated
            scores (means and standard errors).
    """
    # Initial logging message
    logger.info(f"Finished evaluation of {model_id} on {dataset_name}.")

    # Initialise the total dict
    total_dict = dict()

    # Logging of the aggregated scores
    for metric_cfg in metric_configs:
        agg_scores = aggregate_scores(scores=scores, metric_config=metric_cfg)
        test_score, test_se = agg_scores["test"]

        msg = f"{metric_cfg.pretty_name}:\n  - Test: {test_score:.2f} ± {test_se:.2f}"

        # Store the aggregated test scores
        total_dict[metric_cfg.name] = test_score
        total_dict[f"{metric_cfg.name}_se"] = test_se

        # Log the scores
        logger.info(msg)

    # Define a dict with both the raw scores and the aggregated scores
    all_scores = dict(raw=scores, total=total_dict)

    # Return the extended scores
    return all_scores


def aggregate_scores(
    scores: Sequence[Dict[str, float]], metric_config: MetricConfig
) -> Dict[str, Tuple[float, float]]:
    """Helper function to compute the mean with confidence intervals.

    Args:
        scores (list):
            List of dictionaries with the names of the metrics as keys, of the form
            "<metric_name>", such as "f1", and values the metric values.
        metric_config (MetricConfig):
            The configuration of the metric, which is used to collect the correct
            metric from `scores`.

    Returns:
        dict:
            Dictionary with key 'test', with values being a pair of floats, containing
            the score and the radius of its 95% confidence interval.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = dict()
        test_scores = [dct[metric_config.name] for dct in scores]
        test_score = np.mean(test_scores)
        if len(test_scores) > 1:
            sample_std = np.std(test_scores, ddof=1)
            test_se = sample_std / np.sqrt(len(test_scores))
        else:
            test_se = np.nan
        results["test"] = (test_score, 1.96 * test_se)

        return results


def internet_connection_available() -> bool:
    try:
        requests.get("https://www.google.com")
        return True
    except RequestException:
        return False
