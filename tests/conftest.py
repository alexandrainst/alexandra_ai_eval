"""Global fixtures for unit tests."""

import pytest

from src.aiai_eval.config import EvaluationConfig
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
