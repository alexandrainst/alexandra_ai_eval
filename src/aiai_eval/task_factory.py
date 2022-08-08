"""Factory that produces tasks from a task configuration."""

from ast import Return
from typing import Type, Union

from .config import DatasetTask, EvaluationConfig


class TaskFactory:
    """Factory which produces tasks from a configuration.

    Args:
        evaluation_config (EvaluationConfig):
            The benchmark configuration to be used in all tasks constructed.

    Attributes:
        evaluation_config (EvaluationConfig):
            The benchmark configuration to be used in all tasks constructed.
    """

    def __init__(self, evaluation_config: EvaluationConfig):
        self.evaluation_config = evaluation_config

    def build_dataset(self, dataset_task: Union[str, DatasetTask]) -> None:
        """Build a dataset from a configuration or a name.
        Args:
            dataset (str or DatasetConfig):
                The name of the dataset, or the dataset configuration.
        Returns:
            Implement BenchmarkDataset:
                The benchmark dataset.
        """
        # TODO: implement BenchmarkDataset analog
        return None
