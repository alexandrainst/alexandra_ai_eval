"""Unit tests for the `named_entity_recognition` module."""

from functools import partial

import pytest
from datasets import Dataset, load_dataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorForTokenClassification

from src.aiai_eval.exceptions import InvalidEvaluation
from src.aiai_eval.named_entity_recognition import NamedEntityRecognition
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


class TestPreprocessDataTransformer:
    @pytest.fixture(scope="class")
    def preprocessed(self, dataset, ner, tokenizer, model_config):
        yield ner._preprocess_data_transformer(
            dataset=dataset,
            framework="pytorch",
            tokenizer=tokenizer,
            config=model_config,
        )

    def test_spacy_framework_throws_exception(self, dataset, ner, tokenizer):
        with pytest.raises(InvalidEvaluation):
            ner._preprocess_data_transformer(
                dataset=dataset,
                framework="spacy",
                tokenizer=tokenizer,
            )

    def test_preprocessed_is_dataset(self, preprocessed):
        assert isinstance(preprocessed, Dataset)

    def test_preprocessed_columns(self, preprocessed):
        assert set(preprocessed.features.keys()) == set(
            [
                "labels",
                "input_ids",
                "token_type_ids",
                "attention_mask",
            ]
        )


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
        assert set(tokenised_dataset.features.keys()) == set(
            [
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
            ]
        )


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
