"""Unit tests for the `gui` module."""

import gradio as gr
import pytest
from click.types import BOOL

from aiai_eval.gui import evaluate, main


class TestEvaluate:
    @pytest.fixture(scope="class")
    def invalid_model_id(self):
        yield "invalid-model-id"

    def test_raise_error_on_invalid_model(self, invalid_model_id, task_config):
        with pytest.raises(gr.Error):
            evaluate(model_id=invalid_model_id, task=task_config.name)


class TestCLI:
    @pytest.fixture(scope="class")
    def params(self):
        ctx = main.make_context(None, list())
        yield {p.name: p.type for p in main.get_params(ctx)}

    def test_cli_param_names(self, params):
        assert set(params.keys()) == {"cache_examples", "help"}

    def test_cli_param_types(self, params):
        assert params["cache_examples"] == BOOL
        assert params["help"] == BOOL
