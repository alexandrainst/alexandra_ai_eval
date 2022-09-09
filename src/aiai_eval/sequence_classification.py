"""Class for sequence classification tasks."""

from functools import partial
from typing import List

from datasets.arrow_dataset import Dataset
from spacy.language import Language
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .exceptions import FrameworkCannotHandleTask, MissingLabel, WrongFeatureColumnName
from .task import Task


class SequenceClassification(Task):
    """Sequence classification task.

    Args:
        task_config (TaskConfig):
            The configuration of the task.
        evaluation_config (EvaluationConfig):
            The configuration of the evaluation.

    Attributes:
        task_config (TaskConfig):
            The configuration of the task.
        evaluation_config (EvaluationConfig):
            The configuration of the evaluation.
    """

    def _preprocess_data(self, dataset: Dataset, framework: str, **kwargs) -> Dataset:

        # If the model is a spaCy model then raise an error, since we have not yet
        # implemented sequence classification evaluation for spaCy models.
        if framework == "spacy":
            raise FrameworkCannotHandleTask(
                framework="spaCy", task=self.task_config.pretty_name
            )

        # Define the tokenization function
        tokenize_fn = partial(
            tokenize,
            tokenizer=kwargs["tokenizer"],
            feature_column_names=self.task_config.feature_column_names,
            label_column_name=self.task_config.label_column_name,
        )

        # Tokenize the samples
        tokenized = dataset.map(tokenize_fn, batched=True)

        # Translate labels to ids
        numericalize_fn = partial(
            create_numerical_labels,
            model_label2id=kwargs["model_config"].label2id,
        )
        preprocessed = tokenized.map(
            numericalize_fn,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return preprocessed

    def _load_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> DataCollator:
        return DataCollatorWithPadding(tokenizer, padding="longest")

    def _get_spacy_predictions(
        self, model: Language, prepared_dataset: Dataset, batch_size: int
    ) -> list:
        raise FrameworkCannotHandleTask(
            framework="spaCy", task=self.task_config.pretty_name
        )

    def _check_if_model_is_trained_for_task(self, model_predictions: list) -> bool:
        sample_preds = model_predictions[0]
        elements_are_floats = isinstance(sample_preds[0], float)
        return elements_are_floats


def tokenize(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    feature_column_names: List[str],
    label_column_name: str,
) -> dict:
    """Tokenize the text in the examples.

    Args:
        examples (dict):
            The examples to tokenize.
        tokenizer (PreTrainedTokenizerBase):
            The tokenizer to use.
        feature_column_names (list of str):
            The names of the columns containing the features.
        label_column_name (str):
            The name of the column containing the labels.

    Returns:
        dict:
            The tokenized examples.

    Raises:
        WrongFeatureColumnName:
            If the feature column names were not found.
    """
    try:
        tokenized_examples = tokenizer(
            *[examples[feat_col] for feat_col in feature_column_names],
            truncation=True,
            padding=True,
        )
        tokenized_examples["labels"] = examples[label_column_name]
        return tokenized_examples

    except KeyError:
        raise WrongFeatureColumnName(feature_column_names)


def create_numerical_labels(examples: dict, model_label2id: dict) -> dict:
    """Creates numerical labels for an example.

    Args:
        examples (dict):
            The examples to create numerical labels for.
        model_label2id (dict):
            The mapping from model labels to ids.

    Returns:
        dict:
            The examples with numerical labels.

    Raises:
        MissingLabel:
            If the label was not found in the model's label2id mapping.
    """
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

    # Return the examples, now with numerical labels
    return examples
