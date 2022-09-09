"""Unit tests for the `task_factory` module."""

import pytest

from src.aiai_eval.config import LabelConfig
from src.aiai_eval.task_factory import TaskFactory
from src.aiai_eval.utils import kebab_to_pascal


@pytest.fixture(scope="module")
def label():
    yield LabelConfig(name="label_name", synonyms=["synonym1", "synonym2"])


@pytest.fixture(scope="module")
def task_factory(evaluation_config):
    yield TaskFactory(evaluation_config)


def test_attributes_correspond_to_arguments(task_factory, evaluation_config):
    assert task_factory.evaluation_config == evaluation_config


def test_configs_are_preserved(task_config, task_factory, evaluation_config):
    task = task_factory.build_task(task_name_or_config=task_config)
    assert task.evaluation_config == evaluation_config
    assert task.task_config == task_config


def test_build_task(task_config, task_factory):
    task = task_factory.build_task(task_name_or_config=task_config)
    assert type(task).__name__ in [
        kebab_to_pascal(task_config.name),
        kebab_to_pascal(task_config.supertask),
    ]
