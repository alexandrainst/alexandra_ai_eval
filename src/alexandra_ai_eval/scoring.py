"""Functions related to scoring."""

import logging
import warnings

import numpy as np

from .config import MetricConfig

logger = logging.getLogger(__name__)


def log_scores(
    task_name: str,
    metric_configs: list[MetricConfig],
    scores: list[dict[str, float]],
    model_id: str,
    only_return_log: bool = False,
    model_type: str = "other",
) -> dict | str:
    """Log the scores.

    Args:
        task_name:
            Name of the dataset.
        metric_configs:
            list of metrics to log.
        scores:
            The scores that are to be logged.
        model_id:
            The full Hugging Face Hub path to the pretrained transformer model.
        only_return_log:
            If only the logging string should be returned. Defaults to False.
        model_type:
            The type of model, either "huggingface" or "other". Defaults to "other".

    Returns:
        If the `only_return_log` is set then a string is returned containing the logged
        evaluation results. Otherwise, a nested dictionary of the evaluation results.
        The keys are the names of the datasets, with values being new dictionaries
        having the model IDs as keys.
    """
    # Initial logging message
    logger.info(f"Finished evaluation of {model_id} on {task_name}.")

    # Initialise the total dict
    total_dict = dict()
    logging_strings: list[str] = list()

    # Logging of the aggregated scores
    for metric_cfg in metric_configs:
        test_score, test_se = aggregate_scores(scores=scores, metric_config=metric_cfg)
        test_score_str = metric_cfg.postprocessing_fn(test_score)
        test_se_str = metric_cfg.postprocessing_fn(test_se)

        msg = f"{metric_cfg.pretty_name}:\n↳  {test_score_str} ± {test_se_str}"
        logging_strings.append(msg)

        # Store the aggregated test scores
        total_dict[metric_cfg.name] = test_score
        total_dict[f"{metric_cfg.name}_se"] = test_se

        # Log the scores
        logger.info(msg)

    # Define a dict with both the raw scores and the aggregated scores
    all_scores = dict(raw=scores, total=total_dict, model_type=model_type)

    # Return the extended scores
    if only_return_log:
        return "\n\n".join(logging_strings)
    else:
        return all_scores


def aggregate_scores(
    scores: list[dict[str, float]], metric_config: MetricConfig
) -> tuple[float, float]:
    """Helper function to compute the mean with confidence intervals.

    Args:
        scores:
            list of dictionaries with the names of the metrics as keys, of the form
            "<metric_name>", such as "f1", and values the metric values.
        metric_config:
            The configuration of the metric, which is used to collect the correct
            metric from `scores`.

    Returns:
        A pair (score, se) of the mean and the standard error of the scores.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_scores = [dct[metric_config.name] for dct in scores]
        test_score = np.mean(test_scores)
        if len(test_scores) > 1:
            sample_std = np.std(test_scores, ddof=1)
            test_se = sample_std / np.sqrt(len(test_scores))
        else:
            test_se = np.nan
        return (float(test_score), float(1.96 * test_se))
