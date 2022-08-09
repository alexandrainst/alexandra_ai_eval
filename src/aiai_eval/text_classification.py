"""Class for text classification tasks."""
from .task import EvaluationDataset


class SentimentAnalysis(EvaluationDataset):
    """Sentiment analysis evaluation dataset.
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


class OffensiveSpeechClassification(EvaluationDataset):
    """Offensive Speech classification evaluation dataset.
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
