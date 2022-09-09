"""Unit tests for the `model_loading` module."""

from copy import deepcopy

import pytest
import torch
from spacy.language import Language

from src.aiai_eval.exceptions import InvalidFramework
from src.aiai_eval.model_loading import load_model


def test_load_model(model_configs, task_config, evaluation_config):
    for model_config in model_configs:
        model = load_model(
            model_config=model_config,
            task_config=task_config,
            evaluation_config=evaluation_config,
        )["model"]
        if model_config.framework == "pytorch":
            assert isinstance(model, torch.nn.Module)
        elif model_config.framework == "spacy":
            assert isinstance(model, Language)


def test_invalid_model_framework(model_configs, task_config, evaluation_config):
    for model_config in model_configs:
        model_config_copy = deepcopy(model_config)
        model_config_copy.framework = "wrong"
        with pytest.raises(InvalidFramework):
            load_model(
                model_config=model_config_copy,
                task_config=task_config,
                evaluation_config=evaluation_config,
            )
