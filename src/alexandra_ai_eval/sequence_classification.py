"""Class for sequence classification tasks."""

import numpy as np
from datasets.arrow_dataset import Dataset
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .config import TaskConfig
from .exceptions import (
    FrameworkCannotHandleTask,
    InvalidEvaluation,
    MissingLabel,
    WrongFeatureColumnName,
)
from .task import Task
from .utils import has_floats


class SequenceClassification(Task):
    """Sequence classification task.

    Args:
        task_config:
            The configuration of the task.
        evaluation_config:
            The configuration of the evaluation.

    Attributes:
        task_config:
            The configuration of the task.
        evaluation_config:
            The configuration of the evaluation.
    """

    def _pytorch_preprocess_fn(
        self,
        examples: BatchEncoding,
        tokenizer: PreTrainedTokenizerBase,
        model_config: PretrainedConfig,
        task_config: TaskConfig,
    ) -> BatchEncoding:
        return tokenize_and_numericalize(
            examples=examples,
            tokenizer=tokenizer,
            feature_column_names=task_config.feature_column_names,
            label_column_name=task_config.label_column_name,
            model_label2id=model_config.label2id,
        )

    def _prepare_predictions_and_labels(
        self,
        predictions: list,
        dataset: Dataset,
        prepared_dataset: Dataset,
        **kwargs,
    ) -> list[tuple[list, list]]:
        # Collapse the logits into single predictions for every sample
        if has_floats(predictions):
            predictions = np.argmax(predictions, axis=-1)

        labels = prepared_dataset["labels"]

        return [(list(predictions), list(labels))]

    def _load_data_collator(
        self, tokenizer_or_processor: PreTrainedTokenizerBase
    ) -> DataCollatorWithPadding:
        return DataCollatorWithPadding(tokenizer_or_processor, padding="longest")

    def _check_if_model_is_trained_for_task(self, model_predictions: list) -> bool:
        sample_preds = model_predictions[0]
        elements_are_floats = sample_preds[0].dtype.kind == "f"
        return elements_are_floats

    def _spacy_preprocess_fn(self, examples: dict) -> dict:
        raise FrameworkCannotHandleTask(
            framework="spaCy", task=self.task_config.pretty_name
        )

    def _extract_spacy_predictions(self, tokens_processed: tuple) -> list:
        raise FrameworkCannotHandleTask(
            framework="spaCy", task=self.task_config.pretty_name
        )


def tokenize_and_numericalize(
    examples: BatchEncoding,
    tokenizer: PreTrainedTokenizerBase,
    feature_column_names: list[str],
    label_column_name: str,
    model_label2id: dict | None,
) -> BatchEncoding:
    """Tokenize and numericalize the text in the examples.

    Args:
        examples:
            The examples to tokenize.
        tokenizer:
            The tokenizer to use.
        feature_column_names:
            The names of the columns containing the features.
        label_column_name:
            The name of the column containing the labels.
        model_label2id:
            The mapping from model labels to ids. If None, the mapping is not set and
            an error will be raised.

    Returns:
        The tokenized and numericalized examples.

    Raises:
        InvalidEvaluation:
            If the model label2id mapping is not set.
        WrongFeatureColumnName:
            If the feature column names were not found.
        MissingLabel:
            If a label in the dataset was not found in the model's label2id mapping.
    """
    if model_label2id is None:
        raise InvalidEvaluation(
            "The model label2id mapping is not set. This is required for evaluation."
        )

    # Attempt to tokenize the examples
    try:
        labels = examples[label_column_name]
        examples = tokenizer(
            *[examples[feat_col] for feat_col in feature_column_names],
            truncation=True,
            padding=True,
        )
        examples["labels"] = labels

    except KeyError:
        raise WrongFeatureColumnName(feature_column_names)

    # Attempt to numericalize the labels
    try:
        examples["labels"] = [model_label2id[lbl.upper()] for lbl in examples["labels"]]

    # The numericalization fails if the label is not in the model's label2id mapping,
    # in which case we raise a MissingLabel exception
    except KeyError:
        missing_label = [
            lbl.upper()
            for lbl in examples["labels"]
            if lbl.upper() not in model_label2id
        ][0]
        raise MissingLabel(label=missing_label, label2id=model_label2id)

    # The numericalization fails if the label is not a string, in which case we
    # check if it is an integer and if so, we assume it is already numericalized
    except AttributeError:
        if not isinstance(examples["labels"][0], int):
            raise MissingLabel(label=examples["labels"][0], label2id=model_label2id)

    # Return the examples, now with numerical labels
    return examples
