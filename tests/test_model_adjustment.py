"""Unit tests for the `model_adjustment` module."""

from copy import deepcopy

import pytest
from transformers.models.auto.configuration_auto import AutoConfig

from aiai_eval.enums import Framework
from aiai_eval.exceptions import InvalidEvaluation
from aiai_eval.model_adjustment import adjust_model_to_task, alter_classification_layer
from aiai_eval.utils import check_supertask, get_class_by_name


@pytest.fixture(scope="module")
def model_config(model_configs):
    for model_config in model_configs:
        if model_config.framework != Framework.SPACY:
            yield model_config


@pytest.fixture(scope="module")
def model(model_config, task_config, evaluation_config):
    # Load the configuration of the pretrained model
    config = AutoConfig.from_pretrained(
        model_config.model_id,
        revision=model_config.revision,
        use_auth_token=evaluation_config.use_auth_token,
    )

    # Check whether the supertask is a valid one
    supertask = task_config.supertask
    allowed_architectures = (
        task_config.architectures if task_config.architectures else []
    )
    (
        supertask_which_is_architectures,
        allowed_and_checked_architectures,
    ) = check_supertask(
        architectures=config.architectures,
        supertask=supertask,
        allowed_architectures=allowed_architectures,
    )

    # Get the model class associated with the supertask
    if supertask_which_is_architectures:
        model_cls = get_class_by_name(
            class_name=f"auto-model-for-{supertask}",
            module_name="transformers",
        )
    # If the class name is not of the form "auto-model-for-<supertask>" then
    # use fallback "architectures" from config to get the model class
    elif allowed_and_checked_architectures:
        model_cls = get_class_by_name(
            class_name=allowed_and_checked_architectures[0],
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
                old_model_id2label=list(model.config.id2label),
                flat_dataset_synonyms=model_id2label,
                dataset_num_labels=len(model_id2label),
            )
            try:
                new_dim = model.classifier.weight.data.shape[0]
                old_dim = old_model.classifier.weight.data.shape[0]
            except AttributeError:
                new_dim = model.classifier.out_proj.weight.data.shape[0]
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
