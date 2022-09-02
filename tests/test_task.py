"""Unit tests for the `task` module."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict, Metric
from transformers import PreTrainedTokenizerBase

from src.aiai_eval.exceptions import (
    InvalidArchitectureForTask,
    InvalidEvaluation,
    InvalidFramework,
)
from src.aiai_eval.task import Task


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


@pytest.fixture(scope="module")
def task(evaluation_config, task_config):
    return TaskDummy(task_config=task_config, evaluation_config=evaluation_config)


# Allows us to skip tests based on values set in the task fixture.
@pytest.fixture(autouse=True)
def skip_if_not_this_task(request, task):
    if request.node.get_closest_marker("skip_if_not_this_task"):
        if (
            request.node.get_closest_marker("skip_if_not_this_task").args[0]
            != task.task_config.name
        ):
            pytest.skip("skipped on this task: {}".format(task.task_config.name))


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


@pytest.mark.parametrize(
    argnames="prediction_type,label_type,use_logits,id2label_active",
    argvalues=[
        (pred_type, label_type, use_logits, id2label_active)
        for pred_type in ["numpy", "torch"]
        for label_type in ["numpy", "torch"]
        for use_logits in [True, False]
        for id2label_active in [True, False]
    ],
)
@pytest.mark.skip_if_not_this_task("sent")
def test_compute_metrics_sent(
    prediction_type,
    label_type,
    use_logits,
    id2label_active,
    task,
):

    # Define logits, labels and id2label
    logits = [[1.0, 2.0, -3.0], [4.0, 5.0, -6.0], [7.0, 1.0, -9.0]]
    labels = [1, 2, 2]
    id2label = [1, 2, 3] if id2label_active else None

    # Set up predictions as an array
    if use_logits and prediction_type == "numpy":
        predictions = np.asarray(logits)
    elif use_logits and prediction_type == "torch":
        predictions = torch.tensor(logits)
    elif not use_logits and prediction_type == "numpy":
        predictions = np.asarray(logits).argmax(axis=1)
    else:
        predictions = torch.tensor(logits).argmax(dim=1)

    # Define labels
    if label_type == "numpy":
        labels = np.array(labels)
    else:
        labels = torch.tensor(labels)

    # Compute metrics
    metrics = task._compute_metrics(
        predictions=predictions,
        labels=labels,
        id2label=id2label,
    )

    # Check metrics
    assert isinstance(metrics, dict)
    for value in metrics.values():
        assert isinstance(value, float)


@pytest.mark.parametrize(
    argnames="id2label_active",
    argvalues=[True, False],
)
@pytest.mark.skip_if_not_this_task("ner")
def test_compute_metrics_ner(
    id2label_active,
    task,
):
    # Define logits, labels and id2label
    logits = [
        ["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
        ["B-PER", "I-PER", "O"],
    ]
    labels = [
        ["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
        ["B-PER", "I-PER", "O"],
    ]
    id2label = ["O", "B-PER", "I-PER", "B-MISC", "I-MISC"] if id2label_active else None

    # Set up predictions and labels as arrays
    predictions = np.asarray(logits)
    labels = np.array(labels)

    # Compute metrics
    metrics = task._compute_metrics(
        predictions=predictions,
        labels=labels,
        id2label=id2label,
    )

    # Check metrics
    assert isinstance(metrics, dict)
    for value in metrics.values():
        assert isinstance(value, float)


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
            split_names.add(task.task_config.train_name)
        if task.task_config.val_name:
            split_names.add(task.task_config.val_name)
        if task.task_config.test_name:
            split_names.add(task.task_config.test_name)
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
    def test_wrong_train_name_in_task_config(
        self, task, attribute_name, evaluation_config
    ):
        task_config_copy = deepcopy(task.task_config)
        task_copy = TaskDummy(
            task_config=task_config_copy, evaluation_config=evaluation_config
        )
        setattr(task_config_copy, attribute_name, "wrong")
        with pytest.raises(InvalidEvaluation):
            task_copy._load_data()


class TestLoadModel:
    def test_load_model(self, model_configs, task):
        for model_config in model_configs:
            model = task._load_model(model_config)["model"]
            assert isinstance(model, torch.nn.Module)

    def test_invalid_model_framework(self, model_configs, task):
        for model_config in model_configs:
            model_config_copy = deepcopy(model_config)
            model_config_copy.framework = "wrong"
            with pytest.raises(InvalidFramework):
                task._load_model(model_config_copy)


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
