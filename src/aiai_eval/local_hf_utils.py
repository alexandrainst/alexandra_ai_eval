"""Utility functions related to loading local Hugging Face models."""

from pathlib import Path
from typing import Dict, Union

import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import EvaluationConfig, ModelConfig, TaskConfig
from .enums import Framework
from .exceptions import InvalidEvaluation, InvalidFramework, ModelDoesNotExist
from .model_adjustment import adjust_model_to_task


def load_local_hf_model(
    model_config: ModelConfig,
    task_config: TaskConfig,
    evaluation_config: EvaluationConfig,
) -> Dict[str, Union[PreTrainedModel, PreTrainedTokenizerBase]]:
    """Load a local Hugging Face model from a path.

    Args:
        model_config (ModelConfig):
            The configuration of the model.
        task_config (TaskConfig):
            The configuration of the task.
        evaluation_config (EvaluationConfig):
            The configuration of the evaluation.

    Returns:
        dict:
            A dictionary containing the model and tokenizer.
    """
    # Get the config, which is used to get the name of the model class to use
    config = AutoConfig.from_pretrained(model_config.model_id)
    architecture = config.architectures[0]

    # Get the model class and intialize the model
    model_cls = getattr(transformers, architecture)
    model = model_cls.from_pretrained(model_config.model_id)

    # If the model is a subclass of a RoBERTa model then we have to add a prefix space
    # to the tokens, by the way the model is constructed.
    tokenizer_id = model_config.tokenizer_id
    prefix = "Roberta" in type(model).__name__
    params = dict(use_fast=True, add_prefix_space=prefix)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=model_config.revision,
        use_auth_token=evaluation_config.use_auth_token,
        **params,
    )

    # Set the maximal length of the tokenizer to the model's maximal length. This is
    # required for proper truncation
    if not hasattr(tokenizer, "model_max_length") or tokenizer.model_max_length > 1_000:

        if hasattr(tokenizer, "max_model_input_sizes"):
            all_max_lengths = tokenizer.max_model_input_sizes.values()
            if len(list(all_max_lengths)) > 0:
                min_max_length = min(list(all_max_lengths))
                tokenizer.model_max_length = min_max_length
            else:
                tokenizer.model_max_length = 512
        else:
            tokenizer.model_max_length = 512

    # Set the model to evaluation mode, making its predictions deterministic
    model.eval()

    # Move the model to the specified device
    model.to(evaluation_config.device)

    # Adjust the model to the task
    adjust_model_to_task(
        model=model,
        model_config=model_config,
        task_config=task_config,
    )

    # Return the model and tokenizer as a dict
    return dict(model=model, tokenizer=tokenizer, model_type="other")


def hf_model_exists_locally(model_id: Union[str, Path]) -> bool:
    """Check if a Hugging Face model exists locally.

    Args:
        model_id (str or Path):
            Path to the model folder.

    Returns:
        bool:
            Whether the model exists locally.
    """
    # Ensure that `model_id` is a Path object
    model_id = Path(model_id)

    # Return False if the model folder does not exist
    if not model_id.exists():
        return False

    # Try to load the model config. If this fails, False is returned
    try:
        AutoConfig.from_pretrained(str(model_id))
    except OSError:
        return False

    # Check that a compatible model file exists
    pytorch_model_exists = model_id.glob("*.bin") or model_id.glob("*.pt")
    jax_model_exists = model_id.glob("*.msgpack")

    # If no model file exists, return False
    if not pytorch_model_exists and not jax_model_exists:
        return False

    # Otherwise, if all these checks succeeded, return True
    return True


def get_hf_model_config_locally(model_folder: Union[str, Path]) -> ModelConfig:
    """Get the model configuration from a local Hugging Face model.

    Args:
        model_folder (str or Path):
            Path to the model folder.

    Returns:
        ModelConfig:
            The model configuration.

    Raises:
        InvalidFramework:
            If there is only a TensorFlow model in the model folder.
        ModelDoesNotExist:
            If there is no model in the model folder.
    """
    # Ensure that the model folder is a Path object
    model_folder = Path(model_folder)

    # Get the Hugging Face model config, from which we can get the model's label
    # conversions
    config = AutoConfig.from_pretrained(model_folder)

    # Ensure that the `id2label` conversion is a list
    id2label = config.id2label
    if isinstance(id2label, dict):
        try:
            id2label = [id2label[idx] for idx in range(len(id2label))]
        except KeyError:
            raise InvalidEvaluation(
                "There is a gap in the indexing dictionary of the model."
            )

    # Make all labels upper case
    id2label = [label.upper() for label in id2label]

    # Determine the framework by looking at the file format of the model
    if model_folder.glob("*.bin") or model_folder.glob("*.pt"):
        framework = Framework.PYTORCH
    elif model_folder.glob("*.msgpack"):
        framework = Framework.JAX
    elif model_folder.glob("*.h5"):
        raise InvalidFramework(framework="tensorflow")
    else:
        raise ModelDoesNotExist(model_id=str(model_folder))

    return ModelConfig(
        model_id=str(model_folder),
        tokenizer_id=str(model_folder),
        processor_id=str(model_folder),
        revision="main",
        framework=framework,
        id2label=id2label,
        label2id=config.label2id,
    )
