"""Class for text classification tasks."""

from functools import partial
from typing import Optional

from datasets import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from .exceptions import InvalidEvaluation
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

    def _preprocess_data(self, dataset: Dataset, framework: str, **kwargs) -> Dataset:
        """Preprocess a dataset by tokenizing and aligning the labels.
        
        Args:
            dataset (Hugging Face dataset):
                The dataset to preprocess.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.
                
        Returns:
            Hugging Face dataset: The preprocessed dataset.
        """
        if framework == "spacy":
            raise InvalidEvaluation(
                "Evaluation of text predictions for SpaCy models is not yet "
                "implemented."
            )

        # We are now assuming we are using pytorch
        tokenizer = kwargs["tokenizer"]

        # Tokenizer helper
        def tokenise(examples: dict) -> dict:
            return tokenizer(examples["text"], truncation=True, padding=True)

        # Tokenise
        tokenised = dataset.map(tokenise, batched=True, load_from_cache_file=False)

        # Translate labels to ids
        numericalise = partial(
            self._create_numerical_labels, label2id=kwargs["dataset_task"].label2id
        )
        preprocessed = tokenised.map(
            numericalise, batched=True, load_from_cache_file=False
        )

        # Remove unused column
        return preprocessed.remove_columns(["text"])

    def _create_numerical_labels(self, examples: dict, label2id: dict) -> dict:
        try:
            examples["label"] = [label2id[lbl.upper()] for lbl in examples["label"]]
        except KeyError:
            raise InvalidEvaluation(
                f"One of the labels in the dataset, {examples['label'].upper()}, does "
                f"not occur in the label2id dictionary {label2id}."
            )
        return examples

    def _load_data_collator(self, tokenizer: PreTrainedTokenizerBase):
        """Load the data collator used to prepare samples during evaluation.
        
        Args:
            tokenizer (Hugging Face tokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.
                
        Returns:
            Hugging Face data collator:
                The data collator.
        """
        return DataCollatorWithPadding(tokenizer, padding="longest")

    def _get_spacy_predictions_and_labels(self, model, dataset: Dataset) -> tuple:
        """Get predictions from SpaCy model on dataset.
        Args:
            model (SpaCy model):
                The model.
            dataset (Hugging Face dataset):
                The dataset.
        Returns:
            A pair of arrays:
                The first array contains the probability predictions and the second
                array contains the true labels.
        """
        raise InvalidEvaluation(
            "Evaluation of text classification tasks for SpaCy models is not possible."
        )


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
