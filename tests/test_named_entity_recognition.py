"""Unit tests for the `named_entity_recognition` module."""

from functools import partial

import numpy as np
import pytest
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.aiai_eval.exceptions import InvalidEvaluation
from src.aiai_eval.hf_hub import get_model_config
from src.aiai_eval.model_loading import load_spacy_model
from src.aiai_eval.named_entity_recognition import (
    NamedEntityRecognition,
    tokenize_and_align_labels,
)
from src.aiai_eval.task_configs import NER


@pytest.fixture(scope="module")
def dataset():
    yield load_dataset("dane", split="train")


@pytest.fixture(scope="module")
def ner(evaluation_config):
    yield NamedEntityRecognition(task_config=NER, evaluation_config=evaluation_config)


@pytest.fixture(scope="module")
def tokenizer():
    yield AutoTokenizer.from_pretrained("DaNLP/da-bert-ner")


@pytest.fixture(scope="module")
def model_config():
    config = AutoConfig.from_pretrained("DaNLP/da-bert-ner")
    config.label2id = {lbl.upper(): idx for lbl, idx in config.label2id.items()}
    yield config


@pytest.fixture(scope="module")
def preprocessed_spacy(dataset, ner):
    yield ner._preprocess_data_spacy(
        dataset=dataset,
    )


class TestPreprocessData:
    @pytest.fixture(scope="class")
    def preprocessed(self, dataset, ner, tokenizer, model_config):
        yield ner._preprocess_data(
            dataset=dataset,
            framework="pytorch",
            tokenizer=tokenizer,
            model_config=model_config,
        )

    def test_spacy_framework_throws_exception(self, dataset, ner, tokenizer):
        with pytest.raises(InvalidEvaluation):
            ner._preprocess_data(
                dataset=dataset,
                framework="spacy",
                tokenizer=tokenizer,
            )

    def test_preprocessed_is_dataset(self, preprocessed):
        assert isinstance(preprocessed, Dataset)

    def test_preprocessed_columns(self, preprocessed):
        assert set(preprocessed.features.keys()) == {
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "labels",
        }


class TestTokenizeAndAlignLabels:
    @pytest.fixture(scope="class")
    def tokenised_dataset(self, ner, model_config, tokenizer, dataset):
        map_fn = partial(
            tokenize_and_align_labels,
            tokenizer=tokenizer,
            model_label2id=model_config.label2id,
            dataset_id2label=ner.task_config.id2label,
            label_column_name=ner.task_config.label_column_name,
        )
        yield dataset.map(map_fn, batched=True, load_from_cache_file=False)

    def test_tokenize_and_align_labels_length(self, tokenised_dataset, dataset):
        assert len(tokenised_dataset) == len(dataset)

    def test_tokenize_and_align_labels_columns(self, tokenised_dataset):
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


class TestExtractSpacyPredictions:
    @pytest.fixture(scope="class")
    def batch_size(self):
        yield 2

    @pytest.fixture(scope="class")
    def spacy_predictions(self, spacy_model, batch_size, dataset):
        processed = spacy_model.pipe(
            dataset[NER.feature_column_names[0]], batch_size=batch_size
        )[0]
        tokens = dataset["tokens"][0]
        token_processed = zip(tokens, processed)
        yield ner._extract_spacy_predictions(token_processed)

    def test_preprocessed_spacy_predictions_length(self, preprocessed_spacy, dataset):
        assert len(preprocessed_spacy) == len(dataset)

    def test_preprocessed_spacy_predictions_columns(self, preprocessed_spacy):
        assert set(preprocessed_spacy.features.keys()) == {
            "text",
            "labels",
            "tokens",
            "lemmas",
            "sent_id",
            "tok_ids",
            "pos_tags",
            "morph_tags",
            "dep_ids",
            "dep_labels",
            "ner_tags",
        }


class TestGetSpacyPredictionsAndLabels:
    @pytest.fixture(scope="class")
    def batch_size(self):
        yield 2

    @pytest.fixture(scope="class")
    def preprocessed_spacy(self, preprocessed_spacy, ner, spacy_model, batch_size):
        yield ner._get_spacy_predictions_and_labels(
            model=spacy_model, dataset=preprocessed_spacy, batch_size=batch_size
        )

    def test_preprocessed_spacy_is_tuple(self, preprocessed_spacy):
        assert isinstance(preprocessed_spacy, tuple)

    def test_preprocessed_spacy_predictions_are_list(self, preprocessed_spacy):
        assert isinstance(preprocessed_spacy[0], list)

    def test_preprocessed_spacy_labels_are_list(self, preprocessed_spacy):
        assert isinstance(preprocessed_spacy[1], list)

    def test_preprocessed_spacy_predictions_and_labels_have_same_length(
        self, preprocessed_spacy
    ):
        assert len(preprocessed_spacy[0]) == len(preprocessed_spacy[1])

    def test_preprocessed_spacy_predictions_are_lists_of_lists(
        self, preprocessed_spacy
    ):
        assert isinstance(preprocessed_spacy[0][0], list)

    def test_preprocessed_spacy_labels_are_lists_of_lists(self, preprocessed_spacy):
        assert isinstance(preprocessed_spacy[1][0], list)


class TestPreprocessDataSpacy:
    def test_preprocessed_is_dataset(self, preprocessed_spacy):
        assert isinstance(preprocessed_spacy, Dataset)

    def test_preprocessed_columns(self, preprocessed_spacy):
        assert set(preprocessed_spacy.features.keys()) == {
            "text",
            "labels",
            "tokens",
            "lemmas",
            "sent_id",
            "tok_ids",
            "pos_tags",
            "morph_tags",
            "dep_ids",
            "dep_labels",
            "ner_tags",
        }
