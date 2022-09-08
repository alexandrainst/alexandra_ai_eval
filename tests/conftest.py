"""Global fixtures for unit tests."""

import pytest

from src.aiai_eval.config import EvaluationConfig, MetricConfig
from src.aiai_eval.hf_hub import get_model_config
from src.aiai_eval.model_loading import load_spacy_model
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
    model_id_mapping = {
        "sentiment-classification": ["pin/senda"],
        "named-entity-recognition": ["DaNLP/da-bert-ner"],
        "question-answering": ["deepset/minilm-uncased-squad2"],
    }
    yield [
        get_model_config(model_id=model_id, evaluation_config=evaluation_config)
        for model_id in model_id_mapping[task_config.name]
    ]


@pytest.fixture(scope="session")
def metric_config():
    yield MetricConfig(
        name="metric-name",
        pretty_name="Metric name",
        huggingface_id="metric-id",
        results_key="metric-key",
        postprocessing_fn=lambda x: f"{x:.2f}",
    )


@pytest.fixture(
    scope="session",
    params=[
        "spacy/da_core_news_sm",
        "spacy/en_core_web_sm",
    ],
)
def spacy_model(request, evaluation_config):
    model_config_spacy = get_model_config(
        model_id=request.param, evaluation_config=evaluation_config
    )
    yield load_spacy_model(model_config=model_config_spacy)["model"]
