"""Unit tests for the `evaluator` module."""

from collections import defaultdict
from typing import Dict

import pytest

from src.aiai_eval.evaluator import Evaluator
from src.aiai_eval.exceptions import (
    InvalidArchitectureForTask,
    ModelDoesNotExistOnHuggingFaceHub,
)
from src.aiai_eval.task_configs import NER, SENT
from src.aiai_eval.task_factory import TaskFactory


class TestEvaluator:
    @pytest.fixture(scope="module")
    def evaluator(self):
        yield Evaluator(
            progress_bar=True,
            save_results=False,
            raise_error_on_invalid_model=False,
            cache_dir=".aiai_cache",
            use_auth_token=False,
            verbose=False,
            track_carbon_emissions=False,
            country_iso_code="",
        )

    @pytest.fixture(scope="class")
    def existing_model_id(self):
        yield "bert-base-uncased"

    @pytest.fixture(scope="class")
    def non_existing_model_id(self):
        yield "invalid-model-id"

    def test_evaluator_is_object(self, evaluator):
        assert isinstance(evaluator, Evaluator)

    def test_evaluator_has_attributes(self, evaluator):
        assert hasattr(evaluator, "evaluation_config")
        assert hasattr(evaluator, "evaluation_results")
        assert hasattr(evaluator, "task_factory")

    def test_evaluator_has_results(self, evaluator):
        assert evaluator.evaluation_results is not None

    def test_evaluator_has_config(self, evaluator):
        assert evaluator.evaluation_config is not None

    def test_evaluator_has_task_factory(self, evaluator):
        assert evaluator.task_factory is not None
        assert isinstance(evaluator.task_factory, TaskFactory)

    def test_prepare_model_ids(self, evaluator, existing_model_id):
        model_ids = evaluator._prepare_model_ids([existing_model_id, existing_model_id])
        assert model_ids == [existing_model_id, existing_model_id]
        model_ids = evaluator._prepare_model_ids(existing_model_id)
        assert model_ids == [existing_model_id]

    def test_prepare_task_config(self, evaluator):
        task_config = evaluator._prepare_task_configs(["ner", "sent"])
        assert task_config == [NER, SENT]
        task_config = evaluator._prepare_task_configs("sent")
        assert task_config == [SENT]

    def test_evaluate_single_raise_exception_model_not_found(
        self, evaluator, non_existing_model_id
    ):
        evaluator.evaluation_config.testing = True
        with pytest.raises(ModelDoesNotExistOnHuggingFaceHub):
            evaluator._evaluate_single(
                task_config=NER, model_id=[non_existing_model_id]
            )

    def test_evaluate_single_raise_exception_invalid_task(
        self, evaluator, existing_model_id
    ):
        evaluator.evaluation_config.testing = True
        with pytest.raises(InvalidArchitectureForTask):
            evaluator._evaluate_single(task_config=NER, model_id=existing_model_id)

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
                        {"macro_f1": 0.16666666666666666, "mcc": 0.25},
                        {"macro_f1": 0.6666666666666666, "mcc": 0.25},
                    ],
                    "total": {
                        "macro_f1": 0.41666666666666663,
                        "macro_f1_se": 0.48999999999999994,
                        "mcc": 0.25,
                        "mcc_se": 0.0,
                    },
                },
            ),
        ],
        ids=["sent_pin-senda", "sent_DaNLP-da-bert-tone-sentiment-polarity"],
    )
    def test_evaluate_single(self, evaluator, model_id, task_config, expected_results):
        evaluator.evaluation_config.testing = True
        evaluator._evaluate_single(task_config=task_config, model_id=model_id)
        results = evaluator.evaluation_results[task_config.name][model_id]
        assert expected_results == results

    # TODO: Once we have more than one type of task, this should test combinations of tasks.
    @pytest.mark.parametrize(
        argnames="model_ids, tasks",
        argvalues=[
            (
                ["pin/senda", "DaNLP/da-bert-tone-sentiment-polarity"],
                ["sent", "sent"],
            ),
        ],
        ids=["sent_pin-senda_sent_DaNLP"],
    )
    def test_evaluate(self, evaluator, model_ids, tasks):
        evaluator.evaluation_config.testing = True

        # Get results from evaluate
        evaluator.evaluate(model_id=model_ids, task=tasks)
        pin_results = evaluator.evaluation_results[tasks[0]][model_ids[0]]
        danlp_results = evaluator.evaluation_results[tasks[1]][model_ids[1]]

        # Reset evaluation results
        evaluator.evaluation_results: Dict[str, dict] = defaultdict(dict)

        # Get results from evaluate_single
        evaluator._evaluate_single(task_config=SENT, model_id=model_ids[0])
        evaluator._evaluate_single(task_config=SENT, model_id=model_ids[1])
        pin_results_single = evaluator.evaluation_results[tasks[0]][model_ids[0]]
        danlp_results_single = evaluator.evaluation_results[tasks[1]][model_ids[1]]

        # Check that the results are the same
        assert pin_results_single == pin_results
        assert danlp_results_single == danlp_results
