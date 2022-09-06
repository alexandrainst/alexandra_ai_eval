"""Functions related to the loading of models."""

import subprocess
import warnings
from subprocess import CalledProcessError
from typing import Any, Dict

import spacy
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer

from .config import EvaluationConfig, ModelConfig, TaskConfig
from .exceptions import InvalidEvaluation, InvalidFramework, ModelFetchFailed
from .model_adjustment import adjust_model_to_task
from .utils import check_supertask, is_module_installed

# Ignore warnings from spaCy. This has to be called after the import,
# as the __init__.py file of spaCy sets the warning levels of spaCy
# warning W036
warnings.filterwarnings("ignore", module="spacy*")


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
            A dictionary containing at least the key 'model', with the value being
            the model. Can contain other objects related to the model, such as its
            tokenizer.

    Raises:
        RuntimeError:
            If the framework is not recognized.
    """
    # Ensure that the framework is installed
    from_flax = model_config.framework == "jax"

    # If the framework is JAX then change it to PyTorch, since we will convert
    # JAX models to PyTorch upon download
    if model_config.framework == "jax":
        model_config.framework = "pytorch"

    if model_config.framework == "pytorch":
        return load_pytorch_model(
            model_config=model_config,
            from_flax=from_flax,
            task_config=task_config,
            evaluation_config=evaluation_config,
        )

    elif model_config.framework == "spacy":
        return load_spacy_model(model_config=model_config)

    else:
        raise InvalidFramework(model_config.framework)


def load_pytorch_model(
    model_config: ModelConfig,
    from_flax: bool,
    task_config: TaskConfig,
    evaluation_config: EvaluationConfig,
) -> Dict[str, Any]:
    """Load a PyTorch model.

    Args:
        model_config (ModelConfig):
            The configuration of the model.
        from_flax (bool):
            Whether the model is a Flax model.

    Returns:
        dict:
            A dictionary containing at least the key 'model', with the value being
            the model. Can contain other objects related to the model, such as its
            tokenizer.
    """
    try:
        # Load the configuration of the pretrained model
        config = AutoConfig.from_pretrained(
            model_config.model_id,
            revision=model_config.revision,
            use_auth_token=evaluation_config.use_auth_token,
        )

        # Check whether the supertask is a valid one
        supertask = task_config.supertask
        check_supertask(architectures=config.architectures, supertask=supertask)

        # Get the model class associated with the supertask
        if supertask == "token-classification":
            model_cls = AutoModelForTokenClassification  # type: ignore
        elif supertask == "sequence-classification":
            model_cls = AutoModelForSequenceClassification  # type: ignore
        else:
            raise ValueError(f"The supertask `{supertask}` was not recognised.")

        # Load the model with the correct model class
        model = model_cls.from_pretrained(
            model_config.model_id,
            revision=model_config.revision,
            use_auth_token=evaluation_config.use_auth_token,
            config=config,
            cache_dir=evaluation_config.cache_dir,
            from_flax=from_flax,
        )

    # If an error occured then throw an informative exception
    except (OSError, ValueError):
        raise InvalidEvaluation(
            f"The model {model_config.model_id} either does not have a frameworks "
            "registered, or it is a private model. If it is a private model then "
            "enable the `--use-auth-token` flag and make  sure that you are "
            "logged in to the Hub via the `huggingface-cli login` command."
        )

    # Ensure that the model is compatible with the task
    adjust_model_to_task(
        model=model,
        model_config=model_config,
        task_config=task_config,
    )

    # If the model is a subclass of a RoBERTa model then we have to add a prefix
    # space to the tokens, by the way the model is constructed.
    m_id = model_config.model_id
    prefix = "Roberta" in type(model).__name__
    params = dict(use_fast=True, add_prefix_space=prefix)
    tokenizer = AutoTokenizer.from_pretrained(
        m_id,
        revision=model_config.revision,
        use_auth_token=evaluation_config.use_auth_token,
        **params,
    )

    # Set the maximal length of the tokenizer to the model's maximal length.
    # This is required for proper truncation
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

    return dict(model=model, tokenizer=tokenizer)


def load_spacy_model(model_config: ModelConfig) -> Dict[str, Any]:
    """Load a spaCy model.

    Args:
        model_config (ModelConfig):
            The configuration of the model.

    Returns:
        dict:
            A dictionary containing at least the key 'model', with the value being
            the model. Can contain other objects related to the model, such as its
            tokenizer.
    """
    local_model_id = model_config.model_id.split("/")[-1]

    # Download the model if it has not already been so
    try:
        if not is_module_installed(local_model_id):
            url = (
                f"https://huggingface.co/{model_config.model_id}/resolve/main/"
                f"{local_model_id}-any-py3-none-any.whl"
            )
            subprocess.run(["pip3", "install", url])

    except CalledProcessError as e:
        raise ModelFetchFailed(model_id=local_model_id, error_msg=e.output)

    # Load the model
    try:
        model = spacy.load(local_model_id)
    except OSError as e:
        raise ModelFetchFailed(
            model_id=model_config.model_id,
            error_msg=str(e),
            message=(
                f"Download of {model_config.model_id} failed, with "
                f"the following error message: {str(e)}."
            ),
        )
    return dict(model=model)
