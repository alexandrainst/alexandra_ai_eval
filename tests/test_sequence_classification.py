"""Unit tests for the `sequence_classification` module."""

import numpy as np
import pytest
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding

from aiai_eval.sequence_classification import SequenceClassification
from aiai_eval.task_configs import SENT


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
