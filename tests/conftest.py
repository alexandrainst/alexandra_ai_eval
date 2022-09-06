"""Global fixtures for unit tests."""

import pytest

from src.aiai_eval.config import EvaluationConfig
from src.aiai_eval.hf_hub import get_model_config
from src.aiai_eval.task_configs import get_all_task_configs
from src.aiai_eval.utils import Device


@pytest.fixture(scope="session")
def evaluation_config():
    yield EvaluationConfig(
        raise_error_on_invalid_model=True,
        cache_dir=".aiai_cache",
        use_auth_token=False,
        progress_bar=False,
        save_results=True,
        verbose=True,
        track_carbon_emissions=True,
        country_iso_code="DNK",
        prefer_device=Device.CPU,
        testing=True,
    )


@pytest.fixture(
    scope="session", params=get_all_task_configs().values(), ids=lambda cfg: cfg.name
)
def task_config(request):
    yield request.param


@pytest.fixture(scope="session")
def model_configs(evaluation_config, task_config):
    model_id_mapping = dict(
        sent=["pin/senda"],
        ner=["DaNLP/da-bert-ner"],
        offensive=["DaNLP/da-electra-hatespeech-detection"],
    )
    yield [
        get_model_config(model_id=model_id, evaluation_config=evaluation_config)
        for model_id in model_id_mapping[task_config.name]
    ]
