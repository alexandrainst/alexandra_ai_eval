"""Unit tests for the `evaluator` module."""

import os
from collections import defaultdict
from typing import Dict

import pytest

from aiai_eval.utils import Device
from src.aiai_eval.evaluator import Evaluator
from src.aiai_eval.exceptions import (
    InvalidArchitectureForTask,
    ModelDoesNotExistOnHuggingFaceHub,
)
from src.aiai_eval.task_configs import OFFENSIVE, SENT
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
        with pytest.raises(ModelDoesNotExistOnHuggingFaceHub):
            evaluator._evaluate_single(
                task_config=task_config, model_id=[non_existing_model_id]
            )

    def test_evaluate_single_raise_exception_invalid_task(
        self, evaluator, existing_model_id, task_config
    ):
        with pytest.raises(InvalidArchitectureForTask):
            evaluator._evaluate_single(
                task_config=task_config, model_id=existing_model_id
            )

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
        ],
        ids=[
            "sent_pin-senda",
            "sent_DaNLP-da-bert-tone-sentiment-polarity",
            "offensive_DaNLP-da-electra-hatespeech-detection",
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
            model_id=["pin/senda", "DaNLP/da-bert-tone-sentiment-polarity"],
            task=["sent", "sent"],
        )
        pin_results = evaluator.evaluation_results["sent"]["pin/senda"]
        danlp_results = evaluator.evaluation_results["sent"][
            "DaNLP/da-bert-tone-sentiment-polarity"
        ]

        # Reset evaluation results
        evaluator.evaluation_results: Dict[str, dict] = defaultdict(dict)

        # Get results from evaluate_single
        evaluator._evaluate_single(task_config=SENT, model_id="pin/senda")
        evaluator._evaluate_single(
            task_config=SENT, model_id="DaNLP/da-bert-tone-sentiment-polarity"
        )
        pin_results_single = evaluator.evaluation_results["sent"]["pin/senda"]
        danlp_results_single = evaluator.evaluation_results["sent"][
            "DaNLP/da-bert-tone-sentiment-polarity"
        ]

        # Check that the results are the same
        assert pin_results_single == pin_results
        assert danlp_results_single == danlp_results
