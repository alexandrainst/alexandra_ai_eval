"""Unit tests for the `task_factory` module."""

from copy import deepcopy

import pytest

from alexandra_ai_eval.config import LabelConfig
from alexandra_ai_eval.exceptions import InvalidTask
from alexandra_ai_eval.task_factory import TaskFactory
from alexandra_ai_eval.utils import kebab_to_pascal


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


def test_raise_error_if_unknown_task(task_config, task_factory):
    task_config_copy = deepcopy(task_config)
    task_config_copy.name = "unknown-task"
    task_config_copy.supertask = "unknown-supertask"
    with pytest.raises(InvalidTask):
        task_factory.build_task(task_name_or_config=task_config_copy)
