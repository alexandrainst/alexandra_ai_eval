"""Main loading functions."""

from typing import Any, Dict

from .config import EvaluationConfig, ModelConfig, TaskConfig
from .enums import Framework
from .exceptions import InvalidFramework, ModelDoesNotExist, ModelIsPrivate
from .hf_hub_utils import (
    get_model_config_from_hf_hub,
    load_model_from_hf_hub,
    model_is_private_on_hf_hub,
)
from .spacy_utils import load_spacy_model, model_exists_on_spacy


def load_model(
    model_config: ModelConfig,
    task_config: TaskConfig,
    evaluation_config: EvaluationConfig,
) -> Dict[str, Any]:
    """Load the model.

    Args:
        model_config (ModelConfig):
            The model configuration.
        task_config (TaskConfig):
            The task configuration.
        evaluation_config (EvaluationConfig):
            The evaluation configuration.

    Returns:
        dict:
            A dictionary containing at least the key 'model', with the value being the
            model. Can contain other objects related to the model, such as its
            tokenizer.

    Raises:
        InvalidFramework:
            If the framework is not recognized.
    """
    # Ensure that the framework is installed
    from_flax = model_config.framework == Framework.JAX

    # If the framework is JAX then change it to PyTorch, since we will convert JAX
    # models to PyTorch upon download
    if model_config.framework == Framework.JAX:
        model_config.framework = Framework.PYTORCH

    if model_config.framework == Framework.PYTORCH:
        return load_model_from_hf_hub(
            model_config=model_config,
            from_flax=from_flax,
            task_config=task_config,
            evaluation_config=evaluation_config,
        )

    elif model_config.framework == Framework.SPACY:
        return load_spacy_model(model_id=model_config.model_id)

    else:
        raise InvalidFramework(model_config.framework)


def get_model_config(model_id: str, evaluation_config: EvaluationConfig) -> ModelConfig:
    """Fetches configuration for a model from the Hugging Face Hub.

    Args:
        model_id (str):
            The full Hugging Face Hub ID of the model.
        evaluation_config (EvaluationConfig):
            The configuration of the benchmark.

    Returns:
        ModelConfig:
            The model configuration.

    Raises:
        ModelDoesNotExist:
            If the model id does not exist on the Hugging Face Hub.
        InvalidFramework:
            If the specified framework is not implemented.
    """
    # Check if model exists on Hugging Face Hub or as a spaCy model, as well as
    # checking if the model is private
    model_is_private = model_is_private_on_hf_hub(model_id=model_id)
    model_on_hf_hub = model_is_private is not None
    model_on_spacy = model_exists_on_spacy(model_id=model_id)

    # If it does not exist on Hugging Face Hub or as a spaCy model, raise an error
    if not model_on_hf_hub and not model_on_spacy:
        raise ModelDoesNotExist(model_id=model_id)

    # If it *does* exist on the Hugging Face Hub, but it is private and we have not
    # specified a token, raise an error
    elif model_on_hf_hub and model_is_private and not evaluation_config.use_auth_token:
        raise ModelIsPrivate(model_id=model_id)

    # If it exists as a spaCy model, we return the spaCy model config
    if model_on_spacy:
        return ModelConfig(model_id=model_id, revision="", framework=Framework.SPACY)

    # Otherwise it exists on the Hugging Face Hub, and we attempt to fetch the
    else:
        return get_model_config_from_hf_hub(
            model_id=model_id, evaluation_config=evaluation_config
        )
