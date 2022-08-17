"""Unit tests for the `cli` module."""

import pytest
from click.types import BOOL, INT, STRING, Choice

from src.aiai_eval.cli import evaluate


@pytest.fixture(scope="module")
def params():
    ctx = evaluate.make_context(None, list())
    yield {p.name: p.type for p in evaluate.get_params(ctx)}


def test_cli_param_names(params):
    assert set(params.keys()) == {
        "model_id",
        "task",
        "auth_token",
        "use_auth_token",
        "track_carbon_emissions",
        "country_iso_code",
        "no_progress_bar",
        "no_save_results",
        "raise_error_on_invalid_model",
        "cache_dir",
        "verbose",
        "help",
    }


def test_cli_param_types(params):
    assert params["model_id"] == STRING
    assert isinstance(params["task"], Choice)
    assert params["auth_token"] == STRING
    assert params["use_auth_token"] == BOOL
    assert params["track_carbon_emissions"] == BOOL
    assert params["country_iso_code"] == STRING
    assert params["no_progress_bar"] == BOOL
    assert params["no_save_results"] == BOOL
    assert params["raise_error_on_invalid_model"] == BOOL
    assert params["cache_dir"] == STRING
    assert params["verbose"] == BOOL
    assert params["help"] == BOOL
