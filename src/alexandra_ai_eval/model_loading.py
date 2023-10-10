"""Main loading functions."""

from typing import Any

from .config import EvaluationConfig, ModelConfig, TaskConfig
from .enums import Framework
from .exceptions import InvalidFramework, ModelDoesNotExist, ModelIsPrivate
from .hf_hub_utils import (
    get_model_config_from_hf_hub,
    load_model_from_hf_hub,
    model_exists_on_hf_hub,
    model_is_private_on_hf_hub,
)
from .local_hf_utils import (
    get_hf_model_config_locally,
    hf_model_exists_locally,
    load_local_hf_model,
)
from .local_pytorch_utils import (
    get_pytorch_model_config_locally,
    load_local_pytorch_model,
    pytorch_model_exists_locally,
)
from .spacy_utils import (
    get_model_config_from_spacy,
    load_spacy_model,
    model_exists_on_spacy,
)


def load_model(
    model_config: ModelConfig,
    task_config: TaskConfig,
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    """Load the model.

    Args:
        model_config:
            The model configuration.
        task_config:
            The task configuration.
        evaluation_config:
            The evaluation configuration.

    Returns:
        A dictionary containing at least the key 'model', with the value being the
        model. Can contain other objects related to the model, such as its tokenizer.

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
        model_on_hf_hub = model_exists_on_hf_hub(
            model_id=model_config.model_id,
            use_auth_token=evaluation_config.use_auth_token,
        )
        if model_on_hf_hub:
            return load_model_from_hf_hub(
                model_config=model_config,
                from_flax=from_flax,
                task_config=task_config,
                evaluation_config=evaluation_config,
            )
        elif hf_model_exists_locally(model_id=model_config.model_id):
            return load_local_hf_model(
                model_config=model_config,
                task_config=task_config,
                evaluation_config=evaluation_config,
            )
        elif pytorch_model_exists_locally(model_id=model_config.model_id):
            return load_local_pytorch_model(
                model_config=model_config,
                task_config=task_config,
                evaluation_config=evaluation_config,
            )
        else:
            raise ModelDoesNotExist(model_id=model_config.model_id)

    elif model_config.framework == Framework.SPACY:
        return load_spacy_model(model_id=model_config.model_id)

    else:
        raise InvalidFramework(model_config.framework)


def get_model_config(
    model_id: str, task_config: TaskConfig, evaluation_config: EvaluationConfig
) -> ModelConfig:
    """Fetches configuration for a model.

    Args:
        model_id:
            The ID of the model.
        task_config:
            The task configuration.
        evaluation_config:
            The configuration of the benchmark.

    Returns:
        The model configuration.

    Raises:
        ModelIsPrivate:
            If the model is private and `use_auth_token` has not been set.
        ModelDoesNotExist:
            If the model id does not exist on the Hugging Face Hub.
    """
    # Define variable with authentication token
    auth = evaluation_config.use_auth_token

    # If the model exists on the Hugging Face Hub, then fetch the model config from
    # there
    if model_exists_on_hf_hub(model_id=model_id, use_auth_token=auth):
        # If the model is private and an authentication token has not been provided,
        # raise an error
        if model_is_private_on_hf_hub(model_id=model_id, use_auth_token=auth) and (
            auth is False or auth == ""
        ):
            raise ModelIsPrivate(model_id=model_id)

        # Otherwise, fetch the model configuration from the Hugging Face Hub
        return get_model_config_from_hf_hub(
            model_id=model_id, evaluation_config=evaluation_config
        )

    # Otherwise, if the model exists on Spacy, then fetch the model config from there
    elif model_exists_on_spacy(model_id=model_id):
        return get_model_config_from_spacy(model_id=model_id)

    # Otherwise, if the model exists locally as a Hugging Face model, then fetch the
    # model config from there
    elif hf_model_exists_locally(model_id=model_id):
        return get_hf_model_config_locally(model_folder=model_id)

    # Otherwise, if the model exists locally as a PyTorch model, then fetch the model
    # config from there
    elif pytorch_model_exists_locally(model_id=model_id):
        return get_pytorch_model_config_locally(
            model_folder=model_id,
            dataset_id2label=task_config.id2label,
        )

    # If it does not exist on any of the available model sources, raise an error
    else:
        raise ModelDoesNotExist(model_id=model_id)
