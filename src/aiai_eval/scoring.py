"""Functions related to scoring."""

import logging
import warnings
from typing import Dict, Sequence, Tuple

import numpy as np

from .config import MetricConfig

logger = logging.getLogger(__name__)


def log_scores(
    task_name: str,
    metric_configs: Sequence[MetricConfig],
    scores: Sequence[Dict[str, float]],
    model_id: str,
) -> dict:
    """Log the scores.

    Args:
        task_name (str):
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
    logger.info(f"Finished evaluation of {model_id} on {task_name}.")

    # Initialise the total dict
    total_dict = dict()

    # Logging of the aggregated scores
    for metric_cfg in metric_configs:
        test_score, test_se = aggregate_scores(scores=scores, metric_config=metric_cfg)
        test_score *= 100
        test_se *= 100

        msg = f"{metric_cfg.pretty_name}:\n  - Test: {test_score:.2f} ± {test_se:.2f}"

        # Store the aggregated test scores
        total_dict[f"test_{metric_cfg.name}"] = test_score
        total_dict[f"test_{metric_cfg.name}_se"] = test_se

        # Log the scores
        logger.info(msg)

    # Define a dict with both the raw scores and the aggregated scores
    all_scores = dict(raw=scores, total=total_dict)

    # Return the extended scores
    return all_scores


def aggregate_scores(
    scores: Sequence[Dict[str, float]], metric_config: MetricConfig
) -> Tuple[float, float]:
    """Helper function to compute the mean with confidence intervals.

    Args:
        scores (list):
            List of dictionaries with the names of the metrics as keys, of the form
            "<metric_name>", such as "f1", and values the metric values.
        metric_config (MetricConfig):
            The configuration of the metric, which is used to collect the correct
            metric from `scores`.

    Returns:
        pair of floats:
            A pair (score, se) of the mean and the standard error of the scores.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_scores = [dct[f"test_{metric_config.name}"] for dct in scores]
        test_score = np.mean(test_scores)
        if len(test_scores) > 1:
            sample_std = np.std(test_scores, ddof=1)
            test_se = sample_std / np.sqrt(len(test_scores))
        else:
            test_se = np.nan
        return (float(test_score), float(1.96 * test_se))
