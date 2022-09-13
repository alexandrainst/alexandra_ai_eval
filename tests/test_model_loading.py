"""Unit tests for the `model_loading` module."""

from copy import deepcopy

import pytest

from src.aiai_eval.config import ModelConfig
from src.aiai_eval.enums import Framework
from src.aiai_eval.exceptions import (
    InvalidEvaluation,
    InvalidFramework,
    ModelFetchFailed,
)
from src.aiai_eval.model_loading import load_model, load_pytorch_model, load_spacy_model


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


class TestLoadPytorchModel:
    def test_raise_error_if_supertask_does_not_correspond_to_automodel_type(
        self, evaluation_config, task_config
    ):
        model_config = ModelConfig(
            model_id="saattrupdan/wav2vec2-xls-r-300m-ftspeech",
            revision="main",
            framework=Framework.PYTORCH,
        )
        task_config_copy = deepcopy(task_config)
        task_config_copy.supertask = "wav-2-vec-2-for-c-t-c"
        with pytest.raises(InvalidEvaluation):
            load_pytorch_model(
                model_config=model_config,
                from_flax=False,
                task_config=task_config_copy,
                evaluation_config=evaluation_config,
            )

    def test_raise_error_if_model_not_accessible(self, evaluation_config, task_config):
        model_config = ModelConfig(
            model_id="invalid-model-id",
            revision="main",
            framework=Framework.PYTORCH,
        )
        with pytest.raises(InvalidEvaluation):
            load_pytorch_model(
                model_config=model_config,
                from_flax=False,
                task_config=task_config,
                evaluation_config=evaluation_config,
            )

    def test_output_dict_has_both_model_and_tokenizer(
        self, evaluation_config, task_config, model_configs
    ):
        for model_config in model_configs:
            if model_config.framework == Framework.PYTORCH:
                model_dict = load_pytorch_model(
                    model_config=model_config,
                    from_flax=False,
                    task_config=task_config,
                    evaluation_config=evaluation_config,
                )
                assert "model" in model_dict
                assert "tokenizer" in model_dict


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
