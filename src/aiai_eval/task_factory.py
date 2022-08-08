"""Factory that produces tasks from a task configuration."""

from typing import Type, Union

from .config import EvaluationConfig


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

    # TODO build_dataset(DatasetConfig)
