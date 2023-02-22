"""Unit tests for the `utils` module."""

import gc
import random

import numpy as np
import pytest
import torch
from transformers import AutoModelForSequenceClassification

from alexandra_ai_eval.enums import Framework
from alexandra_ai_eval.exceptions import InvalidArchitectureForTask
from alexandra_ai_eval.utils import (
    check_supertask,
    clear_memory,
    enforce_reproducibility,
    internet_connection_available,
    is_module_installed,
)


class TestEnforceReproducibility:
    def test_random_module(self):
        enforce_reproducibility(framework="random")
        first_random_value = random.random()
        second_random_value = random.random()
        enforce_reproducibility(framework="random")
        third_random_value = random.random()
        assert first_random_value != second_random_value
        assert first_random_value == third_random_value

    def test_numpy_random_module(self):
        enforce_reproducibility(framework="numpy")
        first_random_value = np.random.random()
        second_random_value = np.random.random()
        enforce_reproducibility(framework="numpy")
        third_random_value = np.random.random()
        assert first_random_value != second_random_value
        assert first_random_value == third_random_value

    def test_numpy_generator(self):
        rng = enforce_reproducibility(framework="numpy")
        first_random_value = rng.random()
        second_random_value = rng.random()
        rng = enforce_reproducibility(framework="numpy")
        third_random_value = rng.random()
        assert first_random_value != second_random_value
        assert first_random_value == third_random_value

    def test_pytorch_random_module(self):
        enforce_reproducibility(framework=Framework.PYTORCH)
        first_random_value = torch.rand(1)
        second_random_value = torch.rand(1)
        enforce_reproducibility(framework=Framework.PYTORCH)
        third_random_value = torch.rand(1)
        assert first_random_value != second_random_value
        assert first_random_value == third_random_value

    def test_pytorch_linear_layer(self):
        enforce_reproducibility(framework=Framework.PYTORCH)
        first_layer = torch.nn.Linear(1, 1).weight
        second_layer = torch.nn.Linear(1, 1).weight
        enforce_reproducibility(framework=Framework.PYTORCH)
        third_layer = torch.nn.Linear(1, 1).weight
        assert first_layer != second_layer
        assert first_layer == third_layer

    def test_pytorch_pretrained_with_classification_head(
        self, task_config, model_configs
    ):
        if task_config.name == "sentiment-classification":
            model_id = model_configs[0].model_id

            enforce_reproducibility(framework=Framework.PYTORCH)
            first_layers = [
                layer
                for layer in AutoModelForSequenceClassification.from_pretrained(
                    model_id
                ).classifier.children()
                if hasattr(layer, "weight")
            ]
            for layer in first_layers:
                torch.nn.init.normal_(layer.weight)

            second_layers = [
                layer
                for layer in AutoModelForSequenceClassification.from_pretrained(
                    model_id
                ).classifier.children()
                if hasattr(layer, "weight")
            ]
            for layer in second_layers:
                torch.nn.init.normal_(layer.weight)

            enforce_reproducibility(framework=Framework.PYTORCH)
            third_layers = [
                layer
                for layer in AutoModelForSequenceClassification.from_pretrained(
                    model_id
                ).classifier.children()
                if hasattr(layer, "weight")
            ]
            for layer in third_layers:
                torch.nn.init.normal_(layer.weight)

            for first_layer, second_layer in zip(first_layers, second_layers):
                assert not torch.equal(first_layer.weight, second_layer.weight)

            for first_layer, third_layer in zip(first_layers, third_layers):
                assert torch.equal(first_layer.weight, third_layer.weight)


class TestIsModuleInstalled:
    def test_module_is_installed(self):
        assert is_module_installed("torch")

    def test_module_is_not_installed(self):
        assert not is_module_installed("not-a-module")


def test_internet_connection_available():
    assert internet_connection_available()


def test_clear_memory():
    orig_count = gc.get_count()
    clear_memory()
    assert gc.get_count() < orig_count


@pytest.mark.parametrize(
    argnames="architectures,supertask,allowed_architectures,raises_error",
    argvalues=[
        (
            ["TokenClassification", "SequenceClassification"],
            "token-classification",
            ["token-classification"],
            False,
        ),
        (
            ["TokenClassification", "SequenceClassification"],
            "sequence-classification",
            ["sequence-classification"],
            False,
        ),
        (
            ["TokenClassification", "SequenceClassification"],
            "not-a-supertask",
            ["not-a-supertask"],
            True,
        ),
        (
            ["TokenClassification"],
            "token-classification",
            ["token-classification"],
            False,
        ),
        (
            ["TokenClassification"],
            "sequence-classification",
            ["sequence-classification"],
            True,
        ),
    ],
)
def test_check_supertask(architectures, supertask, allowed_architectures, raises_error):
    if raises_error:
        with pytest.raises(InvalidArchitectureForTask):
            check_supertask(architectures, supertask, allowed_architectures)
    else:
        check_supertask(
            architectures=architectures,
            supertask=supertask,
            allowed_architectures=allowed_architectures,
        )
