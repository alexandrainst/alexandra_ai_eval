"""Factory that produces tasks from a task configuration."""

from typing import Type, Union

from .config import DatasetTask, EvaluationConfig
from .named_entity_recognition import NEREvaluation
from .task import EvaluationTask
from .task_configs import get_all_dataset_tasks
from .text_classification import OffensiveSpeechClassification, SentimentAnalysis


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

    def build_task(self, dataset: Union[str, DatasetTask]) -> EvaluationTask:
        """Build a evaluation task from a configuration or a name.

        Args:
            dataset (str or DatasetTask):
                The name of the dataset, or the dataset configuration.

        Returns:
            dataset (EvaluationTask):
                The evaluation task.
        """
        # Get the dataset configuration
        dataset_task: DatasetTask
        if isinstance(dataset, str):
            name_to_dataset_task = get_all_dataset_tasks()
            dataset_task = name_to_dataset_task[dataset]
        else:
            dataset_task = dataset

        # Get the benchmark class based on the task
        evaluation_cls: Type[EvaluationTask]
        if dataset_task.supertask == "text-classification":
            if dataset_task.name == "sent":
                evaluation_cls = SentimentAnalysis
            elif dataset_task.name == "offensive":
                evaluation_cls = OffensiveSpeechClassification

        elif dataset_task.supertask == "token-classification":
            if dataset_task.name == "ner":
                evaluation_cls = NEREvaluation
        else:
            raise ValueError(f"Unknown dataset task: {dataset_task.supertask}")

        # Create the dataset
        dataset_obj = evaluation_cls(
            dataset_task=dataset_task, evaluation_config=self.evaluation_config
        )

        return dataset_obj
