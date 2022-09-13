"""Unit tests for the `hf_hub` module."""

import pytest

from src.aiai_eval.config import ModelConfig
from src.aiai_eval.enums import Framework
from src.aiai_eval.exceptions import ModelDoesNotExist
from src.aiai_eval.hf_hub import (
    get_model_config,
    model_exists_on_hf_hub,
    model_exists_on_spacy,
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


class TestGetModelConfig:
    @pytest.fixture(
        scope="class",
        params=[
            ("bert-base-uncased", "main", Framework.PYTORCH),
            (
                "saattrupdan/wav2vec2-xls-r-300m-ftspeech@"
                "69d0c52dd895e2c1e21518638039638a8539a9cd",
                "69d0c52dd895e2c1e21518638039638a8539a9cd",
                Framework.PYTORCH,
            ),
            (
                "pin/senda",
                "main",
                Framework.PYTORCH,
            ),
            (
                "en_core_web_sm",
                "",
                Framework.SPACY,
            ),
        ],
        ids=[
            "huggingface-hub",
            "huggingface-hub-with-revision",
            "huggingface-hub-with-organization",
            "spacy",
        ],
    )
    def model_id_revision_framework(self, request):
        yield request.param

    @pytest.fixture(scope="class")
    def model_config(self, model_id_revision_framework, evaluation_config):
        model_id, _, _ = model_id_revision_framework
        yield get_model_config(model_id=model_id, evaluation_config=evaluation_config)

    def test_model_config_is_model_config(self, model_config):
        assert isinstance(model_config, ModelConfig)

    def test_model_config_has_correct_information(
        self, model_config, model_id_revision_framework
    ):
        model_id, revision, framework = model_id_revision_framework
        assert model_config.model_id == model_id.split("@")[0]
        assert model_config.revision == revision
        assert model_config.framework == framework

    def test_invalid_model_id(self, evaluation_config):
        with pytest.raises(ModelDoesNotExist):
            get_model_config(
                model_id="invalid-model-id", evaluation_config=evaluation_config
            )
