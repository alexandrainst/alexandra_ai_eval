"""Unit tests for the `task_factory` module."""

import pytest

from src.aiai_eval.config import EvaluationConfig, Label
from src.aiai_eval.named_entity_recognition import NamedEntityRecognition
from src.aiai_eval.task_configs import NER, SENT
from src.aiai_eval.task_factory import TaskFactory
from src.aiai_eval.text_classification import TextClassification


@pytest.fixture(scope="module")
def evaluation_config():
    yield EvaluationConfig(
        raise_error_on_invalid_model=True,
        cache_dir="cache_dir",
        use_auth_token=True,
        progress_bar=True,
        save_results=True,
        verbose=True,
        testing=True,
    )


@pytest.fixture(scope="module")
def label():
    yield Label(name="label_name", synonyms=["synonym1", "synonym2"])


@pytest.fixture(scope="module")
def task_factory(evaluation_config):
    yield TaskFactory(evaluation_config)


def test_attributes_correspond_to_arguments(task_factory, evaluation_config):
    assert task_factory.evaluation_config == evaluation_config


def test_configs_are_preserved(task_factory, evaluation_config):
    dataset = task_factory.build_task(task_name_or_config=SENT)
    assert dataset.evaluation_config == evaluation_config
    assert dataset.task_config == SENT


def test_build_sent_dataset(task_factory):
    dataset = task_factory.build_task(task_name_or_config=SENT)
    assert isinstance(dataset, TextClassification)


def test_build_ner_dataset(task_factory):
    dataset = task_factory.build_task(task_name_or_config=NER)
    assert isinstance(dataset, NamedEntityRecognition)
