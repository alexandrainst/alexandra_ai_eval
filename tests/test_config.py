"""Unit tests for the `config` module."""

import pytest

from src.aiai_eval.config import (
    EvaluationConfig,
    Label,
    MetricConfig,
    ModelConfig,
    TaskConfig,
)


@pytest.fixture(scope="module")
def metric_config():
    yield MetricConfig(
        name="metric_name",
        pretty_name="Metric name",
        huggingface_id="metric_id",
        results_key="metric_key",
    )


@pytest.fixture(scope="module")
def label():
    yield Label(name="label_name", synonyms=["synonym1", "synonym2"])


@pytest.fixture(scope="class")
def task_config(metric_config, label):
    yield TaskConfig(
        name="task_name",
        pretty_name="Task name",
        huggingface_id="dataset_id",
        supertask="supertask_name",
        metrics=[metric_config],
        labels=[label],
        feature_column_name="column_name",
        train_name="train",
        val_name="val",
        test_name="test",
    )


class TestMetricConfig:
    def test_metric_config_is_object(self, metric_config):
        assert isinstance(metric_config, MetricConfig)

    def test_attributes_correspond_to_arguments(self, metric_config):
        assert metric_config.name == "metric_name"
        assert metric_config.pretty_name == "Metric name"
        assert metric_config.huggingface_id == "metric_id"
        assert metric_config.results_key == "metric_key"

    def test_default_value_of_compute_kwargs(self, metric_config):
        assert metric_config.compute_kwargs == dict()


class TestLabel:
    def test_label_is_object(self, label):
        assert isinstance(label, Label)

    def test_attributes_correspond_to_arguments(self, label):
        assert label.name == "label_name"
        assert label.synonyms == ["synonym1", "synonym2"]


class TestTaskConfig:
    def test_task_config_is_object(self, task_config):
        assert isinstance(task_config, TaskConfig)

    def test_attributes_correspond_to_arguments(
        self, task_config, metric_config, label
    ):
        assert task_config.name == "task_name"
        assert task_config.pretty_name == "Task name"
        assert task_config.huggingface_id == "dataset_id"
        assert task_config.supertask == "supertask_name"
        assert task_config.metrics == [metric_config]
        assert task_config.labels == [label]
        assert task_config.feature_column_name == "column_name"
        assert task_config.train_name == "train"
        assert task_config.val_name == "val"
        assert task_config.test_name == "test"

    def test_id2label(self, task_config, label):
        assert task_config.id2label == [label.name]

    def test_label2id(self, task_config, label):
        assert task_config.label2id == {
            label.name: 0,
            label.synonyms[0]: 0,
            label.synonyms[1]: 0,
        }

    def test_num_labels(self, task_config):
        assert task_config.num_labels == 1

    def test_label_synonyms(self, task_config, label):
        assert task_config.label_synonyms == [
            [
                label.name,
                label.synonyms[0],
                label.synonyms[1],
            ]
        ]


class TestEvaluationConfig:
    def test_evaluation_config_is_object(self, evaluation_config):
        assert isinstance(evaluation_config, EvaluationConfig)

    def test_attributes_correspond_to_arguments(self, evaluation_config):
        assert evaluation_config.raise_error_on_invalid_model is True
        assert evaluation_config.cache_dir == ".aiai_cache"
        assert evaluation_config.use_auth_token is True
        assert evaluation_config.progress_bar is False
        assert evaluation_config.save_results is True
        assert evaluation_config.verbose is True
        assert evaluation_config.testing is True


class TestModelConfig:
    @pytest.fixture(scope="class")
    def model_config(self):
        yield ModelConfig(
            model_id="model_id",
            revision="revision",
            framework="framework",
        )

    def test_model_config_is_object(self, model_config):
        assert isinstance(model_config, ModelConfig)

    def test_attributes_correspond_to_arguments(self, model_config):
        assert model_config.model_id == "model_id"
        assert model_config.revision == "revision"
        assert model_config.framework == "framework"
