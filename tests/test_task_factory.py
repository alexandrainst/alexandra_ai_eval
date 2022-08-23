"""Unit tests for the `task_factory` module."""

import pytest

from src.aiai_eval.config import Label
from src.aiai_eval.named_entity_recognition import NamedEntityRecognition
from src.aiai_eval.task_factory import TaskFactory
from src.aiai_eval.text_classification import TextClassification


@pytest.fixture(scope="module")
def label():
    yield Label(name="label_name", synonyms=["synonym1", "synonym2"])


@pytest.fixture(scope="module")
def task_factory(evaluation_config):
    yield TaskFactory(evaluation_config)


def test_attributes_correspond_to_arguments(task_factory, evaluation_config):
    assert task_factory.evaluation_config == evaluation_config


def test_configs_are_preserved(task_config, task_factory, evaluation_config):
    dataset = task_factory.build_task(task_name_or_config=task_config)
    assert dataset.evaluation_config == evaluation_config
    assert dataset.task_config == task_config


def test_build_dataset(task_config, task_factory):
    dataset = task_factory.build_task(task_name_or_config=task_config)
    if task_config.supertask == "sequence-classification":
        assert isinstance(dataset, TextClassification)
    elif task_config.name == "ner":
        assert isinstance(dataset, NamedEntityRecognition)
