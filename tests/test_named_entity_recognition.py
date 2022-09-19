"""Unit tests for the `named_entity_recognition` module."""

from functools import partial

import numpy as np
import pytest
from datasets.load import load_dataset
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

from aiai_eval.named_entity_recognition import (
    NamedEntityRecognition,
    tokenize_and_align_labels,
)
from aiai_eval.task_configs import NER


@pytest.fixture(scope="module")
def dataset():
    yield load_dataset("dane", split="train")


@pytest.fixture(scope="module")
def ner(evaluation_config):
    yield NamedEntityRecognition(task_config=NER, evaluation_config=evaluation_config)


@pytest.fixture(scope="module")
def tokenizer():
    yield AutoTokenizer.from_pretrained(
        "Maltehb/aelaectra-danish-electra-small-cased-ner-dane"
    )


class TestTokenizeAndAlignLabels:
    @pytest.fixture(scope="class")
    def tokenised_datasets(self, ner, task_config, tokenizer, dataset):
        if task_config == NER:
            all_datasets = list()
            map_fn = partial(
                tokenize_and_align_labels,
                tokenizer=tokenizer,
                model_label2id=ner.task_config.label2id,
                dataset_id2label=ner.task_config.id2label,
                label_column_name=ner.task_config.label_column_name,
            )
            tokenised_dataset = dataset.map(
                map_fn, batched=True, load_from_cache_file=False
            )
            all_datasets.append(tokenised_dataset)
            return all_datasets
        else:
            return None

    def test_tokenize_and_align_labels_length(self, tokenised_datasets, dataset):
        if tokenised_datasets is not None:
            for tokenised_dataset in tokenised_datasets:
                assert len(tokenised_dataset) == len(dataset)

    def test_tokenize_and_align_labels_columns(self, tokenised_datasets):
        if tokenised_datasets is not None:
            for tokenised_dataset in tokenised_datasets:
                assert set(tokenised_dataset.features.keys()) == {
                    "text",
                    "ner_tags",
                    "input_ids",
                    "token_type_ids",
                    "attention_mask",
                    "tokens",
                    "lemmas",
                    "sent_id",
                    "tok_ids",
                    "pos_tags",
                    "morph_tags",
                    "dep_ids",
                    "dep_labels",
                    "labels",
                }


class TestLoadDataCollator:
    @pytest.fixture(scope="class")
    def data_collator(self, ner, tokenizer):
        yield ner._load_data_collator(tokenizer=tokenizer)

    def test_data_collator_dtype(self, data_collator):
        assert isinstance(data_collator, DataCollatorForTokenClassification)

    def test_label_pad_token_id_is_minus_hundred(self, data_collator):
        assert data_collator.label_pad_token_id == -100


def test_compute_metrics(ner):

    # Define predictions and labels
    predictions = [
        ["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
        ["B-PER", "I-PER", "O"],
    ]
    labels = [
        ["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
        ["B-PER", "I-PER", "O"],
    ]

    # Set up predictions and labels as arrays
    predictions_and_labels = [
        (np.asarray(predictions), np.array(labels)),
    ]

    # Compute metrics
    metrics = ner._compute_metrics(
        predictions_and_labels=predictions_and_labels,
    )

    # Check metrics
    assert isinstance(metrics, dict)
    for value in metrics.values():
        assert isinstance(value, float)
