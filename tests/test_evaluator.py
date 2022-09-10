"""Unit tests for the `evaluator` module."""

import os
from collections import defaultdict

import pytest

from aiai_eval.enums import CountryCode, Device
from src.aiai_eval.evaluator import Evaluator
from src.aiai_eval.exceptions import ModelDoesNotExist
from src.aiai_eval.task_factory import TaskFactory


@pytest.fixture(scope="module")
def evaluator():
    evaluator = Evaluator(
        progress_bar=True,
        save_results=False,
        raise_error_on_invalid_model=False,
        cache_dir=".aiai_cache",
        use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        track_carbon_emissions=False,
        country_code=CountryCode.EMPTY,
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

    def test_evaluate_single(
        self, evaluator, model_configs, task_config, model_total_scores
    ):
        for idx, model_config in enumerate(model_configs):
            model_id = model_config.model_id
            evaluator._evaluate_single(task_config=task_config, model_id=model_id)
            results = evaluator.evaluation_results[task_config.name][model_id]
            assert results["total"] == model_total_scores[idx]


class TestEvaluate:
    def test_evaluate_is_identical_to_evaluate_single(
        self, evaluator, task_config, model_configs
    ):
        if len(model_configs) > 1:

            model_ids = [model_config.model_id for model_config in model_configs]

            # Get results from evaluate
            evaluator.evaluate(model_id=model_ids, task=task_config.name)
            evaluate_results = [
                evaluator.evaluation_results[task_config.name][model_id]
                for model_id in model_ids
            ]

            # Reset evaluation results
            evaluator.evaluation_results = defaultdict(dict)

            # Get results from evaluate_single
            for model_id in model_ids:
                evaluator._evaluate_single(task_config=task_config, model_id=model_id)
            evaluate_single_results = [
                evaluator.evaluation_results[task_config.name][model_id]
                for model_id in model_ids
            ]

            # Check that the results are the same
            assert evaluate_results == evaluate_single_results
