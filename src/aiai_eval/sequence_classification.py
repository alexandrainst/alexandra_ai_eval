"""Class for text classification tasks."""

from functools import partial

from datasets.arrow_dataset import Dataset
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .exceptions import InvalidEvaluation, MissingLabel, WrongFeatureColumnName
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
        """Preprocess the data.

        Args:
            dataset (Hugging Face Dataset):
                The dataset to preprocess.
            framework (str):
                The framework used for the model.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.

        Returns:
            Hugging Face Dataset:
                The preprocessed dataset.
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
            try:
                return tokenizer(
                    *[
                        examples[feat_col]
                        for feat_col in self.task_config.feature_column_names
                    ],
                    truncation=True,
                    padding=True,
                )
            except KeyError:
                raise WrongFeatureColumnName(self.task_config.feature_column_names)

        # Tokenise
        tokenised = dataset.map(tokenise, batched=True, load_from_cache_file=False)

        # Translate labels to ids
        numericalise = partial(
            self._create_numerical_labels, label2id=kwargs["config"].label2id
        )
        preprocessed = tokenised.map(
            numericalise, batched=True, load_from_cache_file=False
        )

        # Remove unused columns
        return preprocessed.remove_columns(self.task_config.feature_column_names)

    def _create_numerical_labels(self, examples: dict, label2id: dict) -> dict:
        """Create numerical labels from the labels.

        Args:
            examples (dict):
                The examples to create numerical labels for.
            label2id (dict):
                The mapping from labels to ids.

        Returns:
            dict: The examples with numerical labels.

        Raises:
            MissingLabel:
                If a label is missing in the `label2id` mapping.
        """
        lbl_col = self.task_config.label_column_name
        try:
            examples[lbl_col] = [label2id[lbl.upper()] for lbl in examples[lbl_col]]
        except KeyError:
            missing_label = [
                lbl.upper() for lbl in examples[lbl_col] if lbl.upper() not in label2id
            ][0]
            raise MissingLabel(label=missing_label, label2id=label2id)
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
            "Evaluation of text classification tasks for SpaCy models is currently "
            "not possible."
        )
