"""Unit tests for the `config` module."""

import os

import pytest

from aiai_eval.config import (
    EvaluationConfig,
    LabelConfig,
    MetricConfig,
    ModelConfig,
    TaskConfig,
)
from aiai_eval.enums import CountryCode, Device, Framework, Modality


@pytest.fixture(scope="module")
def label():
    yield LabelConfig(name="label-name", synonyms=["synonym1", "synonym2"])


class TestLabelConfig:
    def test_label_is_object(self, label):
        assert isinstance(label, LabelConfig)

    def test_attributes_correspond_to_arguments(self, label):
        assert label.name == "label-name"
        assert label.synonyms == ["synonym1", "synonym2"]


class TestMetricConfig:
    def test_metric_config_is_object(self, metric_config):
        assert isinstance(metric_config, MetricConfig)

    def test_attributes_correspond_to_arguments(self, metric_config):
        assert metric_config.name == "metric-name"
        assert metric_config.pretty_name == "Metric name"
        assert metric_config.huggingface_id == "metric-id"
        assert metric_config.results_key == "metric-key"
        assert metric_config.postprocessing_fn(10.123456789) == "10.12"

    def test_default_value_of_compute_kwargs(self, metric_config):
        assert metric_config.compute_kwargs == dict()


class TestTaskConfig:
    @pytest.fixture(scope="class")
    def task_config(self, metric_config, label):
        yield TaskConfig(
            name="task-name",
            huggingface_id="dataset-id",
            huggingface_subset=None,
            supertask="supertask-name",
            architectures=["supertask-name"],
            modality=Modality("text"),
            metrics=[metric_config],
            labels=[label],
            feature_column_names=["column-name"],
            label_column_name="label",
            test_name="test",
        )

    def test_task_config_is_object(self, task_config):
        assert isinstance(task_config, TaskConfig)

    def test_attributes_correspond_to_arguments(
        self, task_config, metric_config, label
    ):
        assert task_config.name == "task-name"
        assert task_config.huggingface_id == "dataset-id"
        assert task_config.huggingface_subset is None
        assert task_config.supertask == "supertask-name"
        assert task_config.metrics == [metric_config]
        assert task_config.labels == [label]
        assert task_config.feature_column_names == ["column-name"]
        assert task_config.label_column_name == "label"
        assert task_config.test_name == "test"

    def test_pretty_name(self, task_config):
        assert task_config.pretty_name == "task name"

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
        auth = os.environ.get("HUGGINGFACE_HUB_TOKEN", True)
        assert evaluation_config.raise_error_on_invalid_model is True
        assert evaluation_config.cache_dir == ".aiai_cache"
        assert evaluation_config.use_auth_token == auth
        assert evaluation_config.progress_bar is False
        assert evaluation_config.save_results is False
        assert evaluation_config.verbose is True
        assert evaluation_config.track_carbon_emissions is False
        assert evaluation_config.country_code == CountryCode.DNK
        assert evaluation_config.prefer_device == Device.CPU
        assert evaluation_config.only_return_log is False
        assert evaluation_config.testing is True

    def test_device(self, evaluation_config):
        assert evaluation_config.device == "cpu"


class TestModelConfig:
    @pytest.fixture(scope="class")
    def model_config(self):
        yield ModelConfig(
            model_id="model-id",
            tokenizer_id="tokenizer-id",
            processor_id="processor-id",
            revision="revision",
            framework=Framework.JAX,
            id2label=["label1", "label2"],
        )

    def test_model_config_is_object(self, model_config):
        assert isinstance(model_config, ModelConfig)

    def test_attributes_correspond_to_arguments(self, model_config):
        assert model_config.model_id == "model-id"
        assert model_config.revision == "revision"
        assert model_config.framework == Framework.JAX
