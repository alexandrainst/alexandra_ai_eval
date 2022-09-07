"""Class for text classification tasks."""

from functools import partial

import torch
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
                tokenised_examples = tokenizer(
                    *[
                        examples[feat_col]
                        for feat_col in self.task_config.feature_column_names
                    ],
                    truncation=True,
                    padding=True,
                )
                tokenised_examples["labels"] = examples[
                    self.task_config.label_column_name
                ]
                return tokenised_examples
            except KeyError:
                raise WrongFeatureColumnName(self.task_config.feature_column_names)

        # Tokenise
        tokenised = dataset.map(tokenise, batched=True)

        # Translate labels to ids
        numericalise = partial(
            self._create_numerical_labels,
            model_label2id=kwargs["model_config"].label2id,
        )
        preprocessed = tokenised.map(
            numericalise,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return preprocessed

    def _create_numerical_labels(self, examples: dict, model_label2id: dict) -> dict:
        try:
            examples["labels"] = [
                model_label2id[lbl.upper()] for lbl in examples["labels"]
            ]
        except KeyError:
            missing_label = [
                lbl.upper()
                for lbl in examples["labels"]
                if lbl.upper() not in model_label2id
            ][0]
            raise MissingLabel(label=missing_label, label2id=model_label2id)
        return examples

    def _load_data_collator(self, tokenizer: PreTrainedTokenizerBase):
        return DataCollatorWithPadding(tokenizer, padding="longest")

    def _extract_spacy_predictions(self, tokens_processed: tuple) -> list:
        raise InvalidEvaluation(
            "Evaluation of text classification tasks for SpaCy models is not possible."
        )

    def _get_spacy_predictions_and_labels(
        self, model, dataset: Dataset, batch_size: int
    ) -> tuple:
        raise InvalidEvaluation(
            "Evaluation of text classification tasks for SpaCy models is not possible."
        )

    def _check_if_model_is_trained_for_task(self, model_predictions: list) -> bool:
        """Check if the model is trained for the task.

        Args:
            model_predictions (list):
                The predictions of the model.

        Returns:
            bool:
                True if the model is trained for the task, False otherwise.
        """
        sample_preds = model_predictions[0]
        return isinstance(sample_preds, torch.Tensor) and isinstance(
            sample_preds[0].item(), float
        )
