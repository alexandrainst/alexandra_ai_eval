"""Adjusting a model's configuration, to make it suitable for a task."""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .config import ModelConfig, TaskConfig
from .enums import Framework
from .exceptions import InvalidEvaluation


def adjust_model_to_task(
    model: nn.Module,
    model_config: ModelConfig,
    task_config: TaskConfig,
) -> None:
    """Adjust the model to the task.

    This ensures that the label IDs in the model are consistent with the label IDs in
    the dataset. If there are labels in the dataset which the model has not been
    trained on, then the model's classification layer is extended to include these
    labels.

    Args:
        model (PyTorch Module):
            The model to adjust the label ids of.
        model_config (ModelConfig):
            The model configuration.
        task_config (TaskConfig):
            The task configuration.

    Raises:
        InvalidEvaluation:
            If there is a gap in the indexing dictionary of the model.
    """
    # Define the types of the label conversions
    model_label2id: Optional[dict]
    model_id2label: Optional[Union[dict, list]]

    # Get the `label2id` and `id2label` conversions from the model config
    try:
        model_label2id = {
            lbl.upper(): idx for lbl, idx in model.config.label2id.items()
        }
    except AttributeError:
        model_label2id = None
    try:
        try:
            model_num_labels = len(model.config.id2label)
            if not isinstance(model.config.id2label, list):
                model_id2label = dict(model.config.id2label)
            else:
                model_id2label = model.config.id2label
            model_id2label = [
                model_id2label[idx].upper() for idx in range(model_num_labels)
            ]
        except IndexError:
            raise InvalidEvaluation(
                "There is a gap in the indexing dictionary of the model."
            )
    except AttributeError:
        model_id2label = None

    # If one of `label2id` or `id2label` exists in the model config, then define the
    # other one from it
    if model_label2id is not None and model_id2label is None:
        model_id2label = {idx: lbl.upper() for lbl, idx in model_label2id.items()}
        model_id2label = [model_id2label[idx] for idx in range(len(model_id2label))]
        model.config.id2label = model_id2label
    if model_label2id is None and model_id2label is not None:
        model_label2id = {lbl.upper(): id for id, lbl in enumerate(model_id2label)}
        model.config.label2id = model_label2id

    # If the model does not have `label2id` or `id2label` conversions, then use the
    # defaults
    if model_label2id is None or model_id2label is None:
        model.config.label2id = task_config.label2id
        model.config.id2label = task_config.id2label

    # If the model *does* have conversions, then ensure that it can deal with all the
    # labels in the default conversions. This ensures that we can smoothly deal with
    # labels that the model have not been trained on (it will just always get those
    # labels wrong)
    else:

        # Collect the dataset labels and model labels in the `model_id2label`
        # conversion list
        for label in task_config.id2label:
            syns = [
                syn
                for lst in task_config.label_synonyms
                for syn in lst
                if label.upper() in lst
            ]
            if all([syn not in model_id2label for syn in syns]):
                model_id2label.append(label)

        # Ensure that the model_id2label does not contain duplicates modulo synonyms
        for idx, label in enumerate(model_id2label):
            try:
                canonical_syn = [
                    syn_lst
                    for syn_lst in task_config.label_synonyms
                    if label.upper() in syn_lst
                ][0][0]
                model_id2label[idx] = canonical_syn

            # IndexError appears when the label does not appear within the
            # label_synonyms (i.e. that we added it in the previous step). In this
            # case, we just skip the label.
            except IndexError:
                continue

        # Get the synonyms of all the labels, new ones included
        new_synonyms = list(task_config.label_synonyms)
        flat_old_synonyms = [syn for lst in task_config.label_synonyms for syn in lst]
        new_synonyms += [
            [label.upper()]
            for label in model_id2label
            if label.upper() not in flat_old_synonyms
        ]

        # Add all the synonyms of the labels into the label2id conversion dictionary
        model_label2id = {
            label.upper(): id
            for id, lbl in enumerate(model_id2label)
            for label_syns in new_synonyms
            for label in label_syns
            if lbl.upper() in label_syns
        }

        # Get the old id2label conversion
        old_id2label = [
            model.config.id2label[idx].upper()
            for idx in range(len(model.config.id2label))
        ]

        # Alter the model's classification layer to match the dataset if the model is
        # missing labels
        if (
            len(model_id2label) > len(old_id2label)
            and model_config.framework == Framework.PYTORCH
        ):
            alter_classification_layer(
                model=model,
                model_id2label=model_id2label,
                old_id2label=old_id2label,
                flat_old_synonyms=flat_old_synonyms,
                task_config=task_config,
            )

        # Update the model's own conversions with the new ones
        model.config.id2label = model_id2label
        model.config.label2id = model_label2id


def alter_classification_layer(
    model: nn.Module,
    model_id2label: list,
    old_id2label: list,
    flat_old_synonyms: list,
    task_config: TaskConfig,
) -> None:
    """Alter the classification layer of the model to match the dataset.

    This changes the classification layer in the finetuned model to be consistent with
    all the labels in the dataset. If the model was previously finetuned on a dataset
    which left out a label, say, then that label will be inserted in the model
    architecture here, but without the model ever predicting it. This will allow the
    model to be benchmarked on such datasets, however.

    Note that this only works on classification tasks and only for transformer models.
    This code needs to be rewritten when we add other types of tasks and model types.

    Args:
        model (PyTorch Model):
            The model to alter the classification layer of.
        model_id2label (list):
            The model's label conversion.
        old_id2label (list):
            The old label conversion.
        flat_old_synonyms (list):
            The synonyms of the old labels.
        task_config (TaskConfig):
            The task configuration.

    Raises:
        InvalidEvaluation:
            If the model has not been trained on any of the labels, or synonyms
            thereof, of if it is not a classification model.
    """
    # Count the number of new labels to add to the model
    num_new_labels = len(model_id2label) - len(old_id2label)

    # If *all* the new labels are new and aren't even synonyms of the model's labels,
    # then raise an exception
    if num_new_labels == task_config.num_labels:
        if len(set(flat_old_synonyms).intersection(old_id2label)) == 0:
            raise InvalidEvaluation(
                "The model has not been trained on any of the labels in the "
                "dataset, or synonyms thereof."
            )

    # Load the weights from the model's current classification layer. This handles both
    # the token classification case and the sequence classification case.
    # NOTE: This might need additional cases (or a general solution) when we start
    # dealing with other tasks.
    try:
        clf_weight = model.classifier.weight.data
    except AttributeError:
        try:
            clf_weight = model.classifier.out_proj.weight.data
        except AttributeError:
            raise InvalidEvaluation("Model does not seem to be a classification model.")

    # Create the new weights, which have zeros at all the new entries
    zeros = torch.zeros(num_new_labels, model.config.hidden_size)
    new_clf_weight = torch.cat((clf_weight, zeros), dim=0)
    new_clf_weight = Parameter(new_clf_weight)

    # Create the new classification layer
    new_clf = nn.Linear(model.config.hidden_size, len(model_id2label))

    # Assign the new weights to the new classification layer, and replace the old
    # classification layer with this one
    new_clf.weight = new_clf_weight
    model.classifier = new_clf

    # Update the number of labels the model thinks it has. This is required to
    # avoid exceptions when evaluating
    model.config.num_labels = len(model_id2label)
    model.num_labels = len(model_id2label)
