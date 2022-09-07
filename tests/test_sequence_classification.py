"""Unit tests for the `sequence_classification` module."""

from copy import deepcopy

import numpy as np
import pytest
from datasets import Dataset, load_dataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding

from src.aiai_eval.exceptions import (
    InvalidEvaluation,
    MissingLabel,
    WrongFeatureColumnName,
)
from src.aiai_eval.sequence_classification import SequenceClassification
from src.aiai_eval.task_configs import SENT


@pytest.fixture(scope="module")
def dataset():
    yield load_dataset("DDSC/angry-tweets", split="train")


@pytest.fixture(scope="module")
def seq_clf(evaluation_config):
    yield SequenceClassification(task_config=SENT, evaluation_config=evaluation_config)


@pytest.fixture(scope="module")
def tokenizer():
    yield AutoTokenizer.from_pretrained("pin/senda")


@pytest.fixture(scope="module")
def model_config():
    config = AutoConfig.from_pretrained("pin/senda")
    config.label2id = {lbl.upper(): idx for lbl, idx in config.label2id.items()}
    yield config


class TestPreprocessData:
    @pytest.fixture(scope="class")
    def preprocessed(self, dataset, seq_clf, tokenizer, model_config):
        yield seq_clf._preprocess_data(
            dataset=dataset,
            framework="pytorch",
            tokenizer=tokenizer,
            model_config=model_config,
        )

    def test_spacy_framework_throws_exception(self, dataset, seq_clf, tokenizer):
        with pytest.raises(InvalidEvaluation):
            seq_clf._preprocess_data(
                dataset=dataset,
                framework="spacy",
                tokenizer=tokenizer,
            )

    def test_preprocessed_is_dataset(self, preprocessed):
        assert isinstance(preprocessed, Dataset)

    def test_preprocessed_columns(self, preprocessed):
        assert list(preprocessed.features.keys()) == [
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "labels",
        ]

    def test_throw_exception_if_feature_column_name_is_wrong(
        self, dataset, evaluation_config, tokenizer, task_config, model_config
    ):
        # Create copy of the sentiment analysis task config, with a wrong feature
        # column name
        sent_cfg_copy = deepcopy(task_config)
        sent_cfg_copy.feature_column_names = "wrong_name"

        # Create a text classification task with the wrong feature column name
        seq_clf_copy = SequenceClassification(
            task_config=sent_cfg_copy, evaluation_config=evaluation_config
        )

        # Attempt to preprocess the dataset with the wrong feature column name
        with pytest.raises(WrongFeatureColumnName):
            seq_clf_copy._preprocess_data(
                dataset=dataset,
                framework="pytorch",
                tokenizer=tokenizer,
                config=model_config,
            )


class TestCreateNumericalLabels:
    @pytest.fixture(scope="class")
    def examples(self):
        yield dict(labels=["POSITIVE", "POSITIVE", "NEGATIVE", "NEUTRAL"])

    @pytest.fixture(scope="class")
    def label2id(self):
        yield dict(NEGATIVE=0, NEUTRAL=1, POSITIVE=2)

    def test_output_is_dict(self, seq_clf, examples, label2id):
        numerical_labels = seq_clf._create_numerical_labels(
            examples=examples, model_label2id=label2id
        )
        assert isinstance(numerical_labels, dict)

    def throw_exception_if_label_is_missing(self, seq_clf, label2id):
        with pytest.raises(MissingLabel):
            seq_clf._create_numerical_labels(
                examples=dict(labels=["not-a-label"]),
                model_label2id=label2id,
            )


class TestLoadDataCollator:
    @pytest.fixture(scope="class")
    def data_collator(self, seq_clf, tokenizer):
        yield seq_clf._load_data_collator(tokenizer=tokenizer)

    def test_data_collator_dtype(self, data_collator):
        assert isinstance(data_collator, DataCollatorWithPadding)

    def test_padding_is_longest(self, data_collator):
        assert data_collator.padding == "longest"


def test_compute_metrics(seq_clf):

    # Define predictions and labels
    predictions_and_labels = [
        (np.array([1, 1, 0]), np.array([1, 2, 2])),
    ]

    # Compute metrics
    metrics = seq_clf._compute_metrics(predictions_and_labels=predictions_and_labels)

    # Check metrics
    assert isinstance(metrics, dict)
    for value in metrics.values():
        assert isinstance(value, float)


def test_get_spacy_predictions_and_labels_raises_exception(seq_clf):
    with pytest.raises(InvalidEvaluation):
        seq_clf._get_spacy_predictions_and_labels(
            model=None, dataset=None, batch_size=None
        )


def test_extract_spacy_predictions_throws_exception(seq_clf):
    with pytest.raises(InvalidEvaluation):
        seq_clf._extract_spacy_predictions(
            tokens_processed=([], []),
        )


def test_get_spacy_predictions_and_labels_throws_exception(seq_clf):
    with pytest.raises(InvalidEvaluation):
        seq_clf._get_spacy_predictions_and_labels(
            model=None, dataset=None, batch_size=None
        )


def test_preprocess_data_spacy_throws_exception(seq_clf):
    with pytest.raises(InvalidEvaluation):
        seq_clf._preprocess_data_spacy(dataset=None)
