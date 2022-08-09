"""Class for named entity recognition tasks."""

from .task import EvaluationTask


class NEREvaluation(EvaluationTask):
    """NER task.

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
