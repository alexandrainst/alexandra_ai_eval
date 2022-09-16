"""Unit tests for the `model_loading` module."""

from copy import deepcopy

import pytest

from src.aiai_eval.exceptions import InvalidFramework, ModelDoesNotExist
from src.aiai_eval.hf_hub_utils import get_model_config_from_hf_hub
from src.aiai_eval.model_loading import get_model_config, load_model
from src.aiai_eval.spacy_utils import get_model_config_from_spacy


class TestLoadModel:
    def test_load_model(self, model_configs, task_config, evaluation_config):
        for model_config in model_configs:
            model = load_model(
                model_config=model_config,
                task_config=task_config,
                evaluation_config=evaluation_config,
            )
            assert model is not None

    def test_raise_error_if_invalid_framework(
        self, model_configs, task_config, evaluation_config
    ):
        for model_config in model_configs:
            model_config_copy = deepcopy(model_config)
            model_config_copy.framework = "invalid-framework"
            with pytest.raises(InvalidFramework):
                load_model(
                    model_config=model_config_copy,
                    task_config=task_config,
                    evaluation_config=evaluation_config,
                )


class TestGetModelConfig:
    @pytest.fixture(scope="class")
    def hf_model_id(self):
        yield "bert-base-uncased"

    @pytest.fixture(scope="class")
    def spacy_model_id(self):
        yield "en_core_web_sm"

    @pytest.fixture(scope="class")
    def non_existing_model_id(self):
        yield "invalid-model-id"

    def test_model_configs_are_the_same_for_hf_models(
        self, hf_model_id, evaluation_config, task_config
    ):
        model_config_1 = get_model_config(
            model_id=hf_model_id,
            task_config=task_config,
            evaluation_config=evaluation_config,
        )
        model_config_2 = get_model_config_from_hf_hub(
            model_id=hf_model_id, evaluation_config=evaluation_config
        )
        assert model_config_1 == model_config_2

    def test_model_configs_are_the_same_for_spacy_models(
        self, spacy_model_id, evaluation_config, task_config
    ):
        model_config_1 = get_model_config(
            model_id=spacy_model_id,
            task_config=task_config,
            evaluation_config=evaluation_config,
        )
        model_config_2 = get_model_config_from_spacy(model_id=spacy_model_id)
        assert model_config_1 == model_config_2

    def test_invalid_model_id(self, evaluation_config, task_config):
        with pytest.raises(ModelDoesNotExist):
            get_model_config(
                model_id="invalid-model-id",
                task_config=task_config,
                evaluation_config=evaluation_config,
            )
