"""Abstract Task class."""

from abc import ABC
from typing import Dict

from datasets import load_metric

from .config import DatasetTask, EvaluationConfig


class EvaluationDataset(ABC):
    """Abstract evaluation dataset class.
    Args:
        dataset_task (DatasetTask):
            The configuration of the dataset task.
        evaluation_config (EvaluationConfig):
            The configuration of the evaluation.
    Attributes:
        dataset_task (DatasetTask):
            The configuration of the dataset task.
        evaluation_config (EvaluationConfig):
            The configuration of the evaluation.
    """

    def __init__(self, dataset_task: DatasetTask, evaluation_config: EvaluationConfig):
        """Initialise the dataset.
        Args:
            dataset_task (DatasetTask):
                The configuration for the dataset.
            evaluation_config (EvaluationConfig):
                The configuration for the benchmark.
        """
        self.dataset_task = dataset_task
        self.evaluation_config = evaluation_config
        self._metrics = {
            metric_cfg.name: load_metric(metric_cfg.huggingface_id)
            for metric_cfg in dataset_task.metrics
        }

    def evaluate(self, model_id: str) -> Dict[str, dict]:
        """Evaluate a model.
        Args:
            model_id (str):
                The full Hugging Face Hub path to the pretrained transformer model. The
                specific model version to use can be added after the suffix '@':
                "model_id@v1.0.0". It can be a branch name, a tag name, or a commit id.
        Returns:
            dict:
                The keys in the dict are 'raw' and 'total', with all the raw scores in
                the first dictionary and the aggregated scores in the second.
        """
        return {model_id: {"foo": "bar"}}

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)
