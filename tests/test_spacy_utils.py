"""Unit tests for the `hf_hub_utils` module."""

import pytest

from src.aiai_eval.enums import Framework
from src.aiai_eval.exceptions import ModelFetchFailed
from src.aiai_eval.spacy_utils import load_spacy_model, model_exists_on_spacy


class TestModelExistsOnSpacy:
    @pytest.fixture(scope="class")
    def existing_model_id(self):
        yield "en_core_web_sm"

    @pytest.fixture(scope="class")
    def non_existing_model_id(self):
        yield "invalid-model-id"

    def test_existing_model_id_exists(self, existing_model_id):
        assert model_exists_on_spacy(existing_model_id)

    def test_non_existing_model_id_does_not_exist(self, non_existing_model_id):
        assert not model_exists_on_spacy(non_existing_model_id)


class TestLoadSpacyModel:
    def test_raise_error_if_model_is_not_available(self):
        with pytest.raises(ModelFetchFailed):
            load_spacy_model(model_id="invalid-model-id")

    def test_output_dict_has_model_but_no_tokenizer(self, model_configs):
        for model_config in model_configs:
            if model_config.framework == Framework.SPACY:
                model_dict = load_spacy_model(model_id=model_config.model_id)
                assert "model" in model_dict
                assert "tokenizer" not in model_dict
