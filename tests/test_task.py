"""Unit tests for the `task` module."""

import numpy as np
import pytest
from datasets.arrow_dataset import Dataset
from datasets.metric import Metric
from spacy.language import Language
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.aiai_eval.task import Task


class TaskDummy(Task):
    """Subclass of Task with dummy values for the abstract methods.

    This class is used to test the methods of Task.
    """

    def _preprocess_data(self, dataset: Dataset, framework: str, **kwargs) -> Dataset:
        return Dataset.from_dict(dict(a=[1, 2, 3], b=[4, 5, 6]))

    def _load_data_collator(self, tokenizer: PreTrainedTokenizerBase):
        return None

    def _get_spacy_predictions(
        self, model: Language, prepared_dataset: Dataset, batch_size: int
    ) -> list:
        return list()

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

    def test_metric_values_are_metrics(self, metrics):
        for metric in metrics.values():
            assert isinstance(metric, Metric)


def test_prepare_predictions_and_labels_output_is_trivial(task):
    predictions = np.array([1, 2, 3])
    dataset = Dataset.from_dict(dict(labels=[1, 2, 2]))
    prepared = task._prepare_predictions_and_labels(
        predictions=predictions,
        dataset=dataset,
        prepared_dataset=dataset,
    )
    np.testing.assert_equal(
        actual=prepared,
        desired=[(predictions, np.array(dataset["labels"]))],
    )


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

    def test_preprocess_data_is_abstract(self, abstract_metods):
        assert "_preprocess_data" in abstract_metods

    def test_load_data_collator_is_abstract(self, abstract_metods):
        assert "_load_data_collator" in abstract_metods

    def test_get_spacy_predictions_and_labels_is_abstract(self, abstract_metods):
        assert "_get_spacy_predictions" in abstract_metods

    def test_check_if_model_is_trained_for_task_is_abstract(self, abstract_metods):
        assert "_check_if_model_is_trained_for_task" in abstract_metods
