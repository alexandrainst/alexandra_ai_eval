"""Abstract Task class."""

from abc import ABC

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
            dataset_config (DatasetConfig):
                The configuration for the dataset.
            evaluation_config (BenchmarkConfig):
                The configuration for the benchmark.
        """
        self.dataset_task = dataset_task
        self.evaluation_config = evaluation_config
        self._metrics = {
            metric_cfg.name: load_metric(metric_cfg.huggingface_id)
            for metric_cfg in dataset_task.metrics
        }
