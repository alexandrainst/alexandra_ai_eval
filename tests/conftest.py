"""Global fixtures for unit tests."""

import os

import pytest

from aiai_eval.config import EvaluationConfig, MetricConfig
from aiai_eval.enums import CountryCode, Device
from aiai_eval.model_loading import get_model_config, load_spacy_model
from aiai_eval.task_configs import get_all_task_configs


@pytest.fixture(scope="session")
def evaluation_config():
    yield EvaluationConfig(
        raise_error_on_invalid_model=True,
        cache_dir=".aiai_cache",
        use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        progress_bar=False,
        save_results=False,
        verbose=True,
        track_carbon_emissions=False,
        country_code=CountryCode.DNK,
        prefer_device=Device.CPU,
        only_return_log=False,
        testing=True,
    )


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
    scope="session", params=get_all_task_configs().values(), ids=lambda cfg: cfg.name
)
def task_config(request):
    yield request.param


@pytest.fixture(scope="session")
def model_configs(evaluation_config, task_config):
    model_id_mapping = {
        "sentiment-classification": ["pin/senda"],
        "discourse-coherence-classification": [
            "alexandrainst/da-discourse-coherence-base"
        ],
        "named-entity-recognition": [
            "Maltehb/aelaectra-danish-electra-small-cased-ner-dane",
            "spacy/da_core_news_sm",
        ],
        "question-answering": ["deepset/minilm-uncased-squad2"],
        "offensive-text-classification": [
            "alexandrainst/da-hatespeech-detection-small"
        ],
        "automatic-speech-recognition": ["openai/whisper-tiny"],
    }
    yield [
        get_model_config(
            model_id=model_id,
            task_config=task_config,
            evaluation_config=evaluation_config,
        )
        for model_id in model_id_mapping[task_config.name]
    ]


@pytest.fixture(scope="session")
def model_total_scores(model_configs):
    score_mapping = {
        "pin/senda": dict(
            macro_f1=1.0,
            macro_f1_se=0.0,
            mcc=1.0,
            mcc_se=0.0,
        ),
        "Maltehb/aelaectra-danish-electra-small-cased-ner-dane": dict(
            micro_f1=0.22222222222222224,
            micro_f1_se=0.4355555555555556,
            micro_f1_no_misc=0.3333333333333333,
            micro_f1_no_misc_se=0.6533333333333333,
        ),
        "spacy/da_core_news_sm": dict(
            micro_f1=0.6857142857142857,
            micro_f1_no_misc=0.75,
            micro_f1_no_misc_se=0.49,
            micro_f1_se=0.22400000000000003,
        ),
        "deepset/minilm-uncased-squad2": dict(
            exact_match=75.0,
            exact_match_se=49.0,
            qa_f1=75.0,
            qa_f1_se=49.0,
        ),
        "alexandrainst/da-hatespeech-detection-small": dict(
            macro_f1=1.0,
            macro_f1_se=0.0,
            mcc=0.0,
            mcc_se=0.0,
        ),
        "alexandrainst/da-discourse-coherence-base": dict(
            macro_f1=0.06666666666666667,
            macro_f1_se=0.13066666666666665,
            mcc=0.0,
            mcc_se=0.0,
        ),
        "openai/whisper-tiny": dict(word_error_rate=3.5, word_error_rate_se=0.0),
    }
    yield [score_mapping[model_cfg.model_id] for model_cfg in model_configs]


@pytest.fixture(scope="session")
def spacy_model(model_configs):
    for model_config in model_configs:
        if model_config.framework == "spacy":
            return load_spacy_model(model_id=model_config.model_id)["model"]
    else:
        return None
