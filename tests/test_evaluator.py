"""Unit tests for the `evaluator` module."""

from collections import defaultdict

import pytest
from requests.exceptions import HTTPError

from alexandra_ai_eval.evaluator import Evaluator
from alexandra_ai_eval.leaderboard_utils import Session
from alexandra_ai_eval.task_factory import TaskFactory


@pytest.fixture(scope="module")
def evaluator(evaluation_config):
    evaluator = Evaluator(
        progress_bar=evaluation_config.progress_bar,
        save_results=evaluation_config.save_results,
        raise_error_on_invalid_model=evaluation_config.raise_error_on_invalid_model,
        cache_dir=evaluation_config.cache_dir,
        use_auth_token=evaluation_config.use_auth_token,
        track_carbon_emissions=evaluation_config.track_carbon_emissions,
        country_code=evaluation_config.country_code,
        prefer_device=evaluation_config.prefer_device,
        only_return_log=evaluation_config.only_return_log,
        verbose=evaluation_config.verbose,
    )
    evaluator.evaluation_config.testing = True
    yield evaluator


@pytest.fixture(scope="module")
def evaluator_invalid_url(evaluation_config):
    evaluator = Evaluator(
        progress_bar=evaluation_config.progress_bar,
        save_results=evaluation_config.save_results,
        leaderboard_url="http://invalid",
        raise_error_on_invalid_model=evaluation_config.raise_error_on_invalid_model,
        cache_dir=evaluation_config.cache_dir,
        use_auth_token=evaluation_config.use_auth_token,
        track_carbon_emissions=evaluation_config.track_carbon_emissions,
        country_code=evaluation_config.country_code,
        prefer_device=evaluation_config.prefer_device,
        only_return_log=evaluation_config.only_return_log,
        verbose=evaluation_config.verbose,
    )
    evaluator.evaluation_config.testing = True
    yield evaluator


class TestEvaluator:
    def test_evaluation_config(self, evaluator, evaluation_config):
        assert evaluator.evaluation_config == evaluation_config

    def test_model_lists_is_none(self, evaluator):
        assert evaluator._model_lists is None

    def test_evaluation_results_is_defaultdict(self, evaluator):
        assert isinstance(evaluator.evaluation_results, defaultdict)

    def test_task_factory_is_object(self, evaluator):
        assert isinstance(evaluator.task_factory, TaskFactory)

    def test_task_factory_evaluation_config_is_the_same(self, evaluator):
        assert evaluator.task_factory.evaluation_config == evaluator.evaluation_config


def test_evaluate_is_identical_to_evaluate_single(
    evaluator, task_config, model_configs
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


class TestPrepareModelIds:
    @pytest.fixture(scope="class")
    def model_id(self):
        yield "bert-base-cased"

    def test_prepare_model_ids_list(self, evaluator, model_id):
        model_ids = evaluator._prepare_model_ids([model_id, model_id])
        assert model_ids == [model_id, model_id]

    def test_prepare_model_ids_str(self, evaluator, model_id):
        model_ids = evaluator._prepare_model_ids(model_id)
        assert model_ids == [model_id]


class TestPrepareTaskConfigs:
    def test_prepare_task_config_list_task(self, evaluator, task_config):
        prepared_task_config = evaluator._prepare_task_configs(
            [task_config.name, task_config.name]
        )
        assert prepared_task_config == [task_config, task_config]

    def test_prepare_task_config_str_task(self, evaluator, task_config):
        prepared_task_config = evaluator._prepare_task_configs(task_config.name)
        assert prepared_task_config == [task_config]


def test_evaluate_single(evaluator, model_configs, task_config, model_total_scores):
    for idx, model_config in enumerate(model_configs):
        model_id = model_config.model_id
        evaluator._evaluate_single(task_config=task_config, model_id=model_id)
        results = evaluator.evaluation_results[task_config.name][model_id]
        assert results["total"] == model_total_scores[idx]


def test_send_results_to_leaderboard(evaluator, model_configs, task_config):
    # Set up evaluator to not send results to leaderboard, so we can test this
    # function
    evaluator.send_results_to_leaderboard = False
    if len(model_configs) > 1:
        model_id = [model_config.model_id for model_config in model_configs][0]

        # Get results from evaluate
        evaluator.evaluate(model_id=model_id, task=task_config.name)

        # Check that the results are sent to leaderboard
        assert all(evaluator._send_results_to_leaderboard())


def test_send_results_to_leaderboard_raises_exception(
    evaluator_invalid_url, model_configs, task_config
):
    if len(model_configs) > 1:
        model_id = [model_config.model_id for model_config in model_configs][0]

        # Get results from evaluate
        with pytest.raises(HTTPError):
            evaluator_invalid_url.evaluate(model_id=model_id, task=task_config.name)
