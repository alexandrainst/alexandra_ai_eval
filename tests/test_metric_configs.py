"""Unit tests for the `metric_configs` module."""

import pytest

from alexandra_ai_eval import metric_configs
from alexandra_ai_eval.config import MetricConfig


@pytest.fixture(scope="module")
def all_object_names():
    yield [
        obj_name
        for obj_name in dir(metric_configs)
        if not obj_name.startswith("_") and obj_name != "MetricConfig"
    ]


def module_contains_only_metric_configs(all_object_names):
    for obj_name in all_object_names:
        obj = getattr(metric_configs, obj_name)
        assert isinstance(obj, MetricConfig)


def module_contains_co2_metrics(all_object_names):
    assert "EMISSIONS" in all_object_names
    assert "POWER" in all_object_names
