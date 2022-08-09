"""Class for text classification tasks."""
from .task import EvaluationTask


class SentimentAnalysis(EvaluationTask):
    """Sentiment analysis evaluation task.

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


class OffensiveSpeechClassification(EvaluationTask):
    """Offensive Speech classification evaluation task.

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
