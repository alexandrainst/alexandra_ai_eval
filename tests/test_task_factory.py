"""Unit tests for the `task_factory` module."""

import pytest

from src.aiai_eval.config import Label
from src.aiai_eval.named_entity_recognition import NamedEntityRecognition
from src.aiai_eval.question_answering import QuestionAnswering
from src.aiai_eval.sequence_classification import SequenceClassification
from src.aiai_eval.task_factory import TaskFactory


@pytest.fixture(scope="module")
def label():
    yield Label(name="label_name", synonyms=["synonym1", "synonym2"])


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
    if task_config.supertask == "sequence-classification":
        assert isinstance(task, SequenceClassification)
    elif task_config.name == "ner":
        assert isinstance(task, NamedEntityRecognition)
    elif task_config.name == "qa":
        assert isinstance(task, QuestionAnswering)
