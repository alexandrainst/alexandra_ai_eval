"""Unit tests for the `evaluator` module."""

import os
from collections import defaultdict

import pytest

from aiai_eval.utils import Device
from src.aiai_eval.evaluator import Evaluator
from src.aiai_eval.exceptions import ModelDoesNotExist
from src.aiai_eval.task_configs import NER, OFFENSIVE, QA, SENT
from src.aiai_eval.task_factory import TaskFactory


@pytest.fixture(scope="module")
def evaluator():
    evaluator = Evaluator(
        progress_bar=True,
        save_results=False,
        raise_error_on_invalid_model=False,
        cache_dir=".aiai_cache",
        use_auth_token=os.environ["HUGGINGFACE_HUB_TOKEN"],
        track_carbon_emissions=False,
        country_iso_code="",
        prefer_device=Device.CPU,
        verbose=False,
    )
    evaluator.evaluation_config.testing = True
    yield evaluator


@pytest.fixture(scope="module")
def non_existing_model_id():
    yield "invalid-model-id"


@pytest.fixture(scope="module")
def existing_model_id():
    yield "bert-base-cased"


class TestPrepareModelIds:
    def test_prepare_model_ids_list(self, evaluator, existing_model_id):
        model_ids = evaluator._prepare_model_ids([existing_model_id, existing_model_id])
        assert model_ids == [existing_model_id, existing_model_id]

    def test_prepare_model_ids_str(self, evaluator, existing_model_id):
        model_ids = evaluator._prepare_model_ids(existing_model_id)
        assert model_ids == [existing_model_id]


class TestPrepareTaskConfig:
    def test_prepare_task_config_list_task(self, evaluator, task_config):
        prepared_task_config = evaluator._prepare_task_configs(
            [task_config.name, task_config.name]
        )
        assert prepared_task_config == [task_config, task_config]

    def test_prepare_task_config_str_task(self, evaluator, task_config):
        prepared_task_config = evaluator._prepare_task_configs(task_config.name)
        assert prepared_task_config == [task_config]


class TestEvaluator:
    def test_evaluator_is_object(self, evaluator):
        assert isinstance(evaluator, Evaluator)

    def test_evaluator_has_attributes_evaluation_config(self, evaluator):
        assert hasattr(evaluator, "evaluation_config")

    def test_evaluator_has_attributes_evaluation_results(self, evaluator):
        assert hasattr(evaluator, "evaluation_results")

    def test_evaluator_has_attributes_evaluation_task_factory(self, evaluator):
        assert hasattr(evaluator, "task_factory")

    def test_evaluator_has_results(self, evaluator):
        assert evaluator.evaluation_results is not None

    def test_evaluator_has_config(self, evaluator):
        assert evaluator.evaluation_config is not None

    def test_evaluator_has_task_factory(self, evaluator):
        assert isinstance(evaluator.task_factory, TaskFactory)


class TestEvaluateSingle:
    def test_evaluate_single_raise_exception_model_not_found(
        self, evaluator, non_existing_model_id, task_config
    ):
        with pytest.raises(ModelDoesNotExist):
            evaluator._evaluate_single(
                task_config=task_config, model_id=non_existing_model_id
            )

    @pytest.mark.parametrize(
        argnames="model_id, task_config, expected_results",
        argvalues=[
            (
                "pin/senda",
                SENT,
                {
                    "raw": [
                        {"macro_f1": 1.0, "mcc": 1.0},
                        {"macro_f1": 1.0, "mcc": 1.0},
                    ],
                    "total": {
                        "macro_f1": 1.0,
                        "macro_f1_se": 0.0,
                        "mcc": 1.0,
                        "mcc_se": 0.0,
                    },
                },
            ),
            (
                "DaNLP/da-bert-tone-sentiment-polarity",
                SENT,
                {
                    "raw": [
                        {"macro_f1": 1.0, "mcc": 1.0},
                        {"macro_f1": 1.0, "mcc": 1.0},
                    ],
                    "total": {
                        "macro_f1": 1.0,
                        "macro_f1_se": 0.0,
                        "mcc": 1.0,
                        "mcc_se": 0.0,
                    },
                },
            ),
            (
                "DaNLP/da-electra-hatespeech-detection",
                OFFENSIVE,
                {
                    "raw": [
                        {"macro_f1": 1.0, "mcc": 0.0},
                        {"macro_f1": 1.0, "mcc": 0.0},
                    ],
                    "total": {
                        "macro_f1": 1.0,
                        "macro_f1_se": 0.0,
                        "mcc": 0.0,
                        "mcc_se": 0.0,
                    },
                },
            ),
            (
                "spacy/da_core_news_md",
                NER,
                {
                    "raw": [
                        {"micro_f1": 0.8, "micro_f1_no_misc": 1.0},
                        {"micro_f1": 0.923076923076923, "micro_f1_no_misc": 1.0},
                    ],
                    "total": {
                        "micro_f1": 0.8615384615384616,
                        "micro_f1_no_misc": 1.0,
                        "micro_f1_no_misc_se": 0.0,
                        "micro_f1_se": 0.12061538461538451,
                    },
                },
            ),
            (
                "Maltehb/aelaectra-danish-electra-small-cased-ner-dane",
                NER,
                {
                    "raw": [
                        {"micro_f1": 0.0, "micro_f1_no_misc": 0.0},
                        {
                            "micro_f1": 0.4444444444444445,
                            "micro_f1_no_misc": 0.6666666666666666,
                        },
                    ],
                    "total": {
                        "micro_f1": 0.22222222222222224,
                        "micro_f1_se": 0.4355555555555556,
                        "micro_f1_no_misc": 0.3333333333333333,
                        "micro_f1_no_misc_se": 0.6533333333333333,
                    },
                },
            ),
            (
                "deepset/minilm-uncased-squad2",
                QA,
                {
                    "raw": [
                        {"exact_match": 100.0, "qa_f1": 100.0},
                        {"exact_match": 50.0, "qa_f1": 50.0},
                    ],
                    "total": {
                        "exact_match": 75.0,
                        "exact_match_se": 49.0,
                        "qa_f1": 75.0,
                        "qa_f1_se": 49.0,
                    },
                },
            ),
        ],
    )
    def test_evaluate_single(self, evaluator, model_id, task_config, expected_results):
        evaluator._evaluate_single(task_config=task_config, model_id=model_id)
        results = evaluator.evaluation_results[task_config.name][model_id]
        assert expected_results == results


class TestEvaluate:
    @pytest.fixture(scope="class")
    def tasks_models(self):
        yield [
            (SENT, "pin/senda", "DaNLP/da-bert-tone-sentiment-polarity"),
            (NER, "DaNLP/da-bert-ner", "saattrupdan/nbailab-base-ner-scandi"),
        ]

    @pytest.mark.parametrize(
        argnames="task_config, model_ids",
        argvalues=[
            (SENT, ["pin/senda", "DaNLP/da-bert-tone-sentiment-polarity"]),
            (NER, ["DaNLP/da-bert-ner", "saattrupdan/nbailab-base-ner-scandi"]),
        ],
        ids=["sent", "ner"],
    )
    def test_evaluate_is_identical_to_evaluate_single(
        self, evaluator, task_config, model_ids
    ):

        # Get results from evaluate
        evaluator.evaluate(model_id=model_ids, task=task_config.name)
        results1 = evaluator.evaluation_results[task_config.name][model_ids[0]]
        results2 = evaluator.evaluation_results[task_config.name][model_ids[1]]

        # Reset evaluation results
        evaluator.evaluation_results = defaultdict(dict)

        # Get results from evaluate_single
        evaluator._evaluate_single(task_config=task_config, model_id=model_ids[0])
        evaluator._evaluate_single(task_config=task_config, model_id=model_ids[1])
        results1_single = evaluator.evaluation_results[task_config.name][model_ids[0]]
        results2_single = evaluator.evaluation_results[task_config.name][model_ids[1]]

        # Check that the results are the same
        assert results1 == results1_single
        assert results2 == results2_single
