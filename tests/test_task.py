"""Unit tests for the `task` module."""

from typing import List, Sequence, Tuple

import pytest
from datasets.arrow_dataset import Dataset
from evaluate import EvaluationModule
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from aiai_eval.config import TaskConfig
from aiai_eval.task import Task


class TaskDummy(Task):
    """Subclass of Task with dummy values for the abstract methods.

    This class is used to test the methods of Task.
    """

    def _prepare_predictions_and_labels(
        self,
        predictions: Sequence,
        dataset: Dataset,
        prepared_dataset: Dataset,
        **kwargs,
    ) -> List[Tuple[list, list]]:
        return list()

    def _spacy_preprocess_fn(self, examples: BatchEncoding) -> BatchEncoding:
        return examples

    def _pytorch_preprocess_fn(
        self,
        examples: BatchEncoding,
        tokenizer: PreTrainedTokenizerBase,
        pytorch_model_config: dict,
        task_config: TaskConfig,
    ) -> BatchEncoding:
        return examples

    def _extract_spacy_predictions(self, tokens_processed: tuple) -> list:
        return list()

    def _load_data_collator(self, tokenizer: PreTrainedTokenizerBase):
        return None

    def _check_if_model_is_trained_for_task(self, model_predictions: list) -> bool:
        return True


@pytest.fixture(scope="session")
def task(evaluation_config, task_config):
    return TaskDummy(task_config=task_config, evaluation_config=evaluation_config)


class TestTaskAttributes:
    @pytest.fixture(scope="class")
    def metrics(self, task):
        yield task._metrics

    def test_metrics_is_dict(self, metrics):
        assert isinstance(metrics, dict)

    def test_metric_keys_are_metric_names(self, metrics, task_config):
        assert set(metrics.keys()) == {cfg.name for cfg in task_config.metrics}

    def test_metric_values_are_evaluator_modules(self, metrics):
        for metric in metrics.values():
            assert isinstance(metric, EvaluationModule)


class TestLoadData:
    @pytest.fixture(scope="class")
    def loaded_data(self, task):
        yield task._load_data()

    def test_loaded_data_is_dataset(self, loaded_data):
        assert isinstance(loaded_data, Dataset)


class TestAbstractMethods:
    @pytest.fixture(scope="class")
    def abstract_metods(self):
        return Task.__abstractmethods__

    def test_prepare_predictions_and_labels_is_abstract(self, abstract_metods):
        assert "_prepare_predictions_and_labels" in abstract_metods

    def test_spacy_preprocess_fn_is_abstract(self, abstract_metods):
        assert "_spacy_preprocess_fn" in abstract_metods

    def test_pytorch_preprocess_fn_is_abstract(self, abstract_metods):
        assert "_pytorch_preprocess_fn" in abstract_metods

    def test_extract_spacy_predictions_is_abstract(self, abstract_metods):
        assert "_extract_spacy_predictions" in abstract_metods

    def test_load_data_collator_is_abstract(self, abstract_metods):
        assert "_load_data_collator" in abstract_metods

    def test_check_if_model_is_trained_for_task_is_abstract(self, abstract_metods):
        assert "_check_if_model_is_trained_for_task" in abstract_metods
