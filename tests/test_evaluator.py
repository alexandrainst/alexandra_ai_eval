"""Unit tests for the `evaluator` module."""

import logging
from collections import defaultdict
from typing import Dict

import pytest

from aiai_eval.utils import Device
from src.aiai_eval.evaluator import Evaluator
from src.aiai_eval.exceptions import ModelDoesNotExist
from src.aiai_eval.task_configs import NER, SENT
from src.aiai_eval.task_factory import TaskFactory


@pytest.fixture(scope="module")
def evaluator():
    evaluator = Evaluator(
        progress_bar=True,
        save_results=False,
        raise_error_on_invalid_model=False,
        cache_dir=".aiai_cache",
        use_auth_token=False,
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

    def test_evaluate_single_raise_warning_invalid_task(
        self, evaluator, existing_model_id, task_config, caplog
    ):
        with caplog.at_level(logging.WARNING):
            evaluator._evaluate_single(
                task_config=task_config, model_id=existing_model_id
            )
        assert (
            f"Skipping evaluation of {existing_model_id} on {task_config.pretty_name} "
            "as the architecture is not supported by the task."
        ) in caplog.text

    @pytest.mark.parametrize(
        argnames="model_id, task_config, expected_results",
        argvalues=[
            (
                "pin/senda",
                SENT,
                {
                    "raw": [
                        {"macro_f1": 1.0, "mcc": 0.5},
                        {"macro_f1": 1.0, "mcc": 0.5},
                    ],
                    "total": {
                        "macro_f1": 1.0,
                        "macro_f1_se": 0.0,
                        "mcc": 0.5,
                        "mcc_se": 0.0,
                    },
                },
            ),
            (
                "DaNLP/da-bert-tone-sentiment-polarity",
                SENT,
                {
                    "raw": [
                        {"macro_f1": 1.0, "mcc": 0.5},
                        {"macro_f1": 1.0, "mcc": 0.5},
                    ],
                    "total": {
                        "macro_f1": 1.0,
                        "macro_f1_se": 0.0,
                        "mcc": 0.5,
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
                "DaNLP/da-bert-ner",
                NER,
                {
                    "raw": [
                        {"micro_f1": 0.75, "micro_f1_no_misc": 1.0},
                        {"micro_f1": 0.8636363636363636, "micro_f1_no_misc": 1.0},
                    ],
                    "total": {
                        "micro_f1": 0.8068181818181819,
                        "micro_f1_no_misc": 1.0,
                        "micro_f1_no_misc_se": 0.0,
                        "micro_f1_se": 0.11136363636363636,
                    },
                },
            ),
        ],
        ids=[
            "sent_pin-senda",
            "sent_DaNLP-da-bert-tone-sentiment-polarity",
            "ner_spacy-da_core_news_md",
            "ner_DaNLP-da-bert-ner",
        ],
    )
    def test_evaluate_single(self, evaluator, model_id, task_config, expected_results):
        evaluator._evaluate_single(task_config=task_config, model_id=model_id)
        results = evaluator.evaluation_results[task_config.name][model_id]
        assert expected_results == results


class TestEvaluate:

    # TODO: Once we have more than one type of task, this should test a combination of tasks,
    # instead of just one type of task.
    def test_evaluate_is_identical_to_evaluate_single(self, evaluator):

        # Get results from evaluate
        evaluator.evaluate(
            model_id=["pin/senda", "DaNLP/da-bert-ner"],
            task=["sent", "ner"],
        )
        pin_results = evaluator.evaluation_results["sent"]["pin/senda"]
        danlp_results = evaluator.evaluation_results["ner"]["DaNLP/da-bert-ner"]

        # Reset evaluation results
        evaluator.evaluation_results: Dict[str, dict] = defaultdict(dict)

        # Get results from evaluate_single
        evaluator._evaluate_single(task_config=SENT, model_id="pin/senda")
        evaluator._evaluate_single(task_config=NER, model_id="DaNLP/da-bert-ner")
        pin_results_single = evaluator.evaluation_results["sent"]["pin/senda"]
        danlp_results_single = evaluator.evaluation_results["ner"]["DaNLP/da-bert-ner"]

        # Check that the results are the same
        assert pin_results_single == pin_results
        assert danlp_results_single == danlp_results
