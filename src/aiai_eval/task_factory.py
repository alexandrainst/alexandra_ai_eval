"""Factory that produces tasks from a task configuration."""

from typing import Type, Union

from .config import EvaluationConfig, TaskConfig
from .named_entity_recognition import NamedEntityRecognition
from .task import Task
from .task_configs import get_all_task_configs
from .text_classification import TextClassification


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

    def build_task(self, task_name_or_config: Union[str, TaskConfig]) -> Task:
        """Build a evaluation task from a configuration or a name.

        Args:
            task_name_or_config (str or TaskConfig):
                The name of the dataset, or the dataset configuration.

        Returns:
            task (Task):
                The evaluation task.
        """
        # Get the dataset configuration
        task_config: TaskConfig
        if isinstance(task_name_or_config, str):
            task_config = get_all_task_configs()[task_name_or_config]
        else:
            task_config = task_name_or_config

        # Get the evaluation class based on the task
        evaluation_cls: Type[Task]
        if task_config.supertask == "text-classification":
            evaluation_cls = TextClassification
        elif task_config.name == "ner":
            evaluation_cls = NamedEntityRecognition
        else:
            raise ValueError(f"Unknown task: {task_config.name}")

        # Create the task
        task_obj = evaluation_cls(
            task_config=task_config, evaluation_config=self.evaluation_config
        )

        return task_obj
