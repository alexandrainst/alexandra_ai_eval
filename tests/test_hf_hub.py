"""Unit tests for the `hf_hub` module."""

import pytest

from src.aiai_eval.config import ModelConfig
from src.aiai_eval.exceptions import ModelDoesNotExistOnHuggingFaceHub
from src.aiai_eval.hf_hub import get_model_config, model_exists_on_hf_hub


class TestGetModelConfig:
    @pytest.fixture(scope="class")
    def model_config(self, evaluation_config):
        yield get_model_config(
            model_id="bert-base-uncased", evaluation_config=evaluation_config
        )

    def test_model_config_is_model_config(self, model_config):
        assert isinstance(model_config, ModelConfig)

    def test_model_config_has_correct_information(self, model_config):
        assert model_config.model_id == "bert-base-uncased"
        assert model_config.revision == "main"
        assert model_config.framework == "pytorch"

    def test_invalid_model_id(self, evaluation_config):
        with pytest.raises(ModelDoesNotExistOnHuggingFaceHub):
            get_model_config(
                model_id="invalid-model-id", evaluation_config=evaluation_config
            )


class TestModelExistsOnHfHub:
    @pytest.fixture(scope="class")
    def existing_model_id(self):
        yield "bert-base-uncased"

    @pytest.fixture(scope="class")
    def non_existing_model_id(self):
        yield "invalid-model-id"

    def test_existing_model_id_exists(self, existing_model_id):
        assert model_exists_on_hf_hub(existing_model_id)

    def test_non_existing_model_id_does_not_exist(self, non_existing_model_id):
        assert not model_exists_on_hf_hub(non_existing_model_id)
