"""Unit tests for the `evaluator` module."""

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

    @pytest.fixture(scope="class")
    def sent(self):
        yield SENT

    @pytest.fixture(scope="class")
    def ner(self):
        yield NER

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

    def test_prepare_task_config(self, evaluator, ner, sent):
        task_config = evaluator._prepare_task_configs(["ner", "sent"])
        assert task_config == [ner, sent]
        task_config = evaluator._prepare_task_configs("sent")
        assert task_config == [sent]

    def test_evaluate_single_raise_exception_model_not_found(
        self, evaluator, non_existing_model_id, ner
    ):
        evaluator.evaluation_config.testing = True
        with pytest.raises(ModelDoesNotExistOnHuggingFaceHub):
            evaluator._evaluate_single(
                task_config=ner, model_id=[non_existing_model_id]
            )

    def test_evaluate_single_raise_exception_invalid_task(
        self, evaluator, existing_model_id, ner
    ):
        evaluator.evaluation_config.testing = True
        with pytest.raises(InvalidArchitectureForTask):
            evaluator._evaluate_single(task_config=ner, model_id=existing_model_id)
