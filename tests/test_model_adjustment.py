"""Unit tests for the `model_adjustment` module."""

from copy import deepcopy

import pytest

from src.aiai_eval.enums import Framework
from src.aiai_eval.exceptions import InvalidEvaluation
from src.aiai_eval.model_adjustment import (
    adjust_model_to_task,
    alter_classification_layer,
)
from src.aiai_eval.utils import get_class_by_name


@pytest.fixture(scope="module")
def model_config(model_configs):
    for model_config in model_configs:
        if model_config.framework != Framework.SPACY:
            yield model_config


@pytest.fixture(scope="module")
def model(model_config, task_config, evaluation_config):
    model_cls = get_class_by_name(
        class_name=f"auto-model-for-{task_config.supertask}",
        module_name="transformers",
    )
    yield model_cls.from_pretrained(  # type: ignore[attr-defined]
        model_config.model_id,
        cache_dir=evaluation_config.cache_dir,
    )


class TestAdjustModelToTask:
    def test_adjusted_model_labels_are_consistent_with_dataset_labels(
        self,
        model,
        model_config,
        task_config,
    ):
        assert not set(task_config.id2label).issubset(set(model.config.id2label))
        adjust_model_to_task(
            model=model,
            model_config=model_config,
            task_config=task_config,
        )
        assert set(task_config.id2label).issubset(set(model.config.id2label))

    def test_raise_error_if_gap_in_model_id2label_dict(
        self,
        model,
        model_config,
        task_config,
    ):
        model.config.id2label = {0: "label1", 2: "label2"}
        with pytest.raises(InvalidEvaluation):
            adjust_model_to_task(
                model=model,
                model_config=model_config,
                task_config=task_config,
            )

    def test_raise_error_if_gap_in_model_id2label_list(
        self,
        model,
        model_config,
        task_config,
    ):
        model.config.id2label = ["label"]
        with pytest.raises(InvalidEvaluation):
            adjust_model_to_task(
                model=model,
                model_config=model_config,
                task_config=task_config,
            )


class TestAlterClassificationLayer:
    def test_no_change_if_model_has_correct_number_of_labels(self, model, task_config):
        if task_config.supertask in {"sequence-classification", "token-classification"}:
            old_model = deepcopy(model)
            alter_classification_layer(
                model=model,
                model_id2label=model.config.id2label,
                old_model_id2label=model.config.id2label,
                flat_dataset_synonyms=model.config.id2label,
                dataset_num_labels=len(model.config.id2label),
            )
            try:
                clf_shape = model.classifier.weight.data.shape
                old_clf_shape = old_model.classifier.weight.data.shape
            except AttributeError:
                clf_shape = model.classifier.out_proj.weight.data.shape
                old_clf_shape = old_model.classifier.out_proj.weight.data.shape
            assert clf_shape == old_clf_shape

    def test_raise_error_if_all_labels_are_new(self, model, task_config):
        if task_config.supertask in {"sequence-classification", "token-classification"}:
            with pytest.raises(InvalidEvaluation):
                alter_classification_layer(
                    model=model,
                    model_id2label=["label1", "label2", "new_label1", "new_label2"],
                    old_model_id2label=["label1", "label2"],
                    flat_dataset_synonyms=["new_label1", "new_label2"],
                    dataset_num_labels=2,
                )

    def test_increase_dimension_if_new_labels(self, model, task_config):
        if task_config.supertask in {"sequence-classification", "token-classification"}:

            if isinstance(model.config.id2label, dict):
                model_id2label = list(model.config.id2label.values()) + ["new_label"]
            else:
                model_id2label = model.config.id2label + ["new_label"]

            old_model = deepcopy(model)
            alter_classification_layer(
                model=model,
                model_id2label=model_id2label,
                old_model_id2label=model.config.id2label,
                flat_dataset_synonyms=model_id2label,
                dataset_num_labels=len(model_id2label),
            )
            try:
                new_dim = model.classifier.weight.data.shape[0]
                old_dim = old_model.classifier.weight.data.shape[0]
            except AttributeError:
                new_dim = model.classifier.weight.data.shape[0]
                old_dim = old_model.classifier.out_proj.weight.data.shape[0]
            assert new_dim == old_dim + 1

    def test_raise_error_if_not_classification_model(self, model, task_config):
        if task_config.supertask not in {
            "sequence-classification",
            "token-classification",
        }:
            with pytest.raises(InvalidEvaluation):
                alter_classification_layer(
                    model=model,
                    model_id2label=["label1", "label2", "new_label1", "new_label2"],
                    old_model_id2label=["label1", "label2"],
                    flat_dataset_synonyms=["new_label1", "new_label2"],
                    dataset_num_labels=2,
                )
