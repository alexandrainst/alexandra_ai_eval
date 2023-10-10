"""Unit tests for the `cli` module."""

import pytest
from click.types import BOOL, STRING, Choice

from alexandra_ai_eval.cli import evaluate


@pytest.fixture(scope="module")
def params():
    ctx = evaluate.make_context(None, list())
    yield {p.name: p.type for p in evaluate.get_params(ctx)}


def test_cli_param_names(params):
    assert set(params.keys()) == {
        "model_id",
        "task",
        "auth_token",
        "token",
        "track_carbon_emissions",
        "country_code",
        "no_progress_bar",
        "no_save_results",
        "raise_error_on_invalid_model",
        "cache_dir",
        "prefer_device",
        "architecture_fname",
        "weight_fname",
        "verbose",
        "help",
    }


def test_cli_param_types(params):
    assert params["model_id"] == STRING
    assert isinstance(params["task"], Choice)
    assert params["auth_token"] == STRING
    assert params["token"] == BOOL
    assert params["track_carbon_emissions"] == BOOL
    assert isinstance(params["country_code"], Choice)
    assert params["no_progress_bar"] == BOOL
    assert params["no_save_results"] == BOOL
    assert params["raise_error_on_invalid_model"] == BOOL
    assert params["cache_dir"] == STRING
    assert isinstance(params["prefer_device"], Choice)
    assert params["verbose"] == BOOL
    assert params["architecture_fname"] == STRING
    assert params["weight_fname"] == STRING
    assert params["help"] == BOOL
