"""Unit tests for the `task` module."""

from copy import deepcopy

import numpy as np
import pytest
from datasets import Dataset, DatasetDict, Metric
from transformers import PreTrainedTokenizerBase

from src.aiai_eval.exceptions import InvalidArchitectureForTask, InvalidEvaluation
from src.aiai_eval.task import Task
from src.aiai_eval.task_configs import SENT, get_all_task_configs


class TaskDummy(Task):
    """Subclass of Task with dummy values for the abstract methods.

    This class is used to test the methods of Task.
    """

    def _preprocess_data_pytorch(self, dataset: Dataset, **kwargs) -> list:
        return list()

    def _preprocess_data_transformer(
        self, dataset: Dataset, framework: str, **kwargs
    ) -> Dataset:
        return Dataset.from_dict(dict(a=[1, 2, 3], b=[4, 5, 6]))

    def _load_data_collator(self, tokenizer: PreTrainedTokenizerBase):
        return None


@pytest.fixture(
    scope="module", params=get_all_task_configs().values(), ids=lambda cfg: cfg.name
)
def task(evaluation_config, request):
    yield TaskDummy(task_config=request.param, evaluation_config=evaluation_config)


class TestTaskAttributes:
    @pytest.fixture(scope="class")
    def metrics(self, task):
        yield task._metrics

    def test_metrics_is_dict(self, metrics):
        assert isinstance(metrics, dict)

    def test_metric_keys_are_metric_names(self, metrics):
        assert set(metrics.keys()) == {cfg.name for cfg in SENT.metrics}

    def test_metric_values_are_metrics(self, metrics):
        for metric in metrics.values():
            assert isinstance(metric, Metric)


class TestEvaluate:
    pass


class TestEvaluatePytorchJax:
    pass


class TestEvaluatePytorchJaxSingleIteration:
    pass


@pytest.mark.skip(reason="Not implemented yet")
class TestEvaluateSpacy:
    pass


class TestComputeMetrics:
    pass


def test_prepare_predictions_and_labels_output_is_trivial(task):
    predictions = np.array([1, 2, 3])
    labels = np.array([1, 2, 3])
    prepared = task._prepare_predictions_and_labels(predictions, labels)
    assert prepared == [(predictions, labels)]


def test_process_data(task):
    dataset_dict = DatasetDict(
        dict(
            train=Dataset.from_dict(dict(a=[1, 2, 3], b=[4, 5, 6])),
            test=Dataset.from_dict(dict(a=[1, 2, 3], b=[4, 5, 6])),
        )
    )
    processed = task._process_data(dataset_dict)
    assert processed == dataset_dict


class TestLoadData:
    @pytest.fixture(scope="class")
    def loaded_data(self, task):
        yield task._load_data()

    def test_loaded_data_is_dataset_dict(self, loaded_data):
        assert isinstance(loaded_data, DatasetDict)

    def test_loaded_data_keys(self, loaded_data, task):
        split_names = set()
        if task.task_config.train_name:
            split_names.add("train")
        if task.task_config.val_name:
            split_names.add("val")
        if task.task_config.test_name:
            split_names.add("test")
        assert set(loaded_data.keys()) == split_names

    def test_loaded_data_values(self, loaded_data):
        for split_name in loaded_data:
            assert isinstance(loaded_data[split_name], Dataset)

    @pytest.mark.parametrize(
        argnames=["attribute_name"],
        argvalues=[
            ("train_name",),
            ("val_name",),
            ("test_name",),
        ],
    )
    def test_wrong_train_name_in_task_config(self, task, attribute_name):
        task_copy = deepcopy(task)
        setattr(task_copy.task_config, attribute_name, "wrong")
        with pytest.raises(InvalidEvaluation):
            task_copy._load_data()


class TestLoadModel:
    pass


class TestLoadPytorchModel:
    pass


class TestLoadSpacyModel:
    pass


@pytest.mark.parametrize(
    argnames="architectures,supertask,raises_error",
    argvalues=[
        (
            ["TokenClassification", "SequenceClassification"],
            "token-classification",
            False,
        ),
        (
            ["TokenClassification", "SequenceClassification"],
            "sequence-classification",
            False,
        ),
        (
            ["TokenClassification", "SequenceClassification"],
            "not-a-supertask",
            True,
        ),
        (
            ["TokenClassification"],
            "token-classification",
            False,
        ),
        (
            ["TokenClassification"],
            "sequence-classification",
            True,
        ),
    ],
)
def test_check_supertask(task, architectures, supertask, raises_error):
    if raises_error:
        with pytest.raises(InvalidArchitectureForTask):
            task._check_supertask(architectures, supertask)
    else:
        task._check_supertask(architectures=architectures, supertask=supertask)


@pytest.mark.skip(reason="Not implemented yet")
class TestAdjustLabelIds:
    pass


class TestAbstractMethods:
    @pytest.fixture(scope="class")
    def abstract_metods(self):
        return Task.__abstractmethods__

    def test_preprocess_data_pytorch_is_abstract(self, abstract_metods):
        assert "_preprocess_data_pytorch" in abstract_metods

    def test_preprocess_data_transformer_is_abstract(self, abstract_metods):
        assert "_preprocess_data_transformer" in abstract_metods

    def test_load_data_collator_is_abstract(self, abstract_metods):
        assert "_load_data_collator" in abstract_metods
