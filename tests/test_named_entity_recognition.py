"""Unit tests for the `named_entity_recognition` module."""

from functools import partial

import pytest
from datasets import Dataset, load_dataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorForTokenClassification

from src.aiai_eval.exceptions import InvalidEvaluation
from src.aiai_eval.hf_hub import get_model_config
from src.aiai_eval.named_entity_recognition import NamedEntityRecognition
from src.aiai_eval.task_configs import NER


@pytest.fixture(scope="module")
def dataset():
    yield load_dataset(
        path="dane",
        split="train",
        download_mode="force_redownload",
    )


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
def model_config_spacy(evaluation_config):
    yield get_model_config("spacy/da_core_news_md", evaluation_config=evaluation_config)


@pytest.fixture(scope="module")
def spacy_model(ner, model_config_spacy):
    yield ner._load_spacy_model(model_config_spacy)["model"]


@pytest.fixture(scope="module")
def preprocessed_spacy(dataset, ner):
    yield ner._preprocess_data_spacy(
        dataset=dataset,
    )


class TestPreprocessDataTransformer:
    @pytest.fixture(scope="class")
    def preprocessed(self, dataset, ner, tokenizer, model_config):
        yield ner._preprocess_data_transformer(
            dataset=dataset,
            framework="pytorch",
            tokenizer=tokenizer,
            config=model_config,
        )

    def test_preprocessed_is_dataset(self, preprocessed):
        assert isinstance(preprocessed, Dataset)

    def test_preprocessed_columns(self, preprocessed):
        assert set(preprocessed.features.keys()) == {
            "labels",
            "input_ids",
            "token_type_ids",
            "attention_mask",
        }


class TestTokenizeAndAlignLabels:
    @pytest.fixture(scope="class")
    def tokenised_dataset(self, ner, model_config, tokenizer, dataset):
        map_fn = partial(
            ner._tokenize_and_align_labels,
            tokenizer=tokenizer,
            label2id=model_config.label2id,
        )
        yield dataset.map(map_fn, batched=True, load_from_cache_file=False)

    def test_tokenize_and_align_labels_length(self, tokenised_dataset, dataset):
        tokenised_dataset = tokenised_dataset.remove_columns(
            [
                "labels",
                "input_ids",
                "token_type_ids",
                "attention_mask",
            ]
        )
        assert len(tokenised_dataset) == len(dataset)

    def test_tokenize_and_align_labels_columns(self, tokenised_dataset):
        assert set(tokenised_dataset.features.keys()) == {
            "text",
            "labels",
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
            "ner_tags",
        }


class TestPreprocessDataPyTorch:
    @pytest.fixture(scope="class")
    def preprocessed(self, dataset, ner, tokenizer, model_config):
        yield ner._preprocess_data_pytorch(
            dataset=dataset,
            tokenizer=tokenizer,
            config=model_config,
        )

    def test_preprocessed_is_list(self, preprocessed):
        assert isinstance(preprocessed, list)


class TestLoadDataCollator:
    @pytest.fixture(scope="class")
    def data_collator(self, ner, tokenizer):
        yield ner._load_data_collator(tokenizer=tokenizer)

    def test_data_collator_dtype(self, data_collator):
        assert isinstance(data_collator, DataCollatorForTokenClassification)

    def test_label_pad_token_id_is_minus_hundred(self, data_collator):
        assert data_collator.label_pad_token_id == -100


class TestExtractSpacyPredictions:
    @pytest.fixture(scope="class")
    def batch_size(self):
        yield 2

    @pytest.fixture(scope="class")
    def spacy_predictions(self, spacy_model, batch_size, dataset):
        processed = spacy_model.pipe(
            dataset[NER.feature_column_name], batch_size=batch_size
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
