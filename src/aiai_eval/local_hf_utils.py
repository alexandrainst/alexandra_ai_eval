"""Utility functions related to loading local Hugging Face models."""

from pathlib import Path
from typing import Dict, Union

import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import ModelConfig
from .enums import Framework


def load_local_hf_model(
    model_config: ModelConfig,
) -> Dict[str, Union[PreTrainedModel, PreTrainedTokenizerBase]]:
    """Load a local Hugging Face model from a path.

    Args:
        model_config (ModelConfig):
            The configuration of the model.

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

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_id)

    # Return the model and tokenizer as a dict
    return dict(model=model, tokenizer=tokenizer)


def hf_model_exists_locally(model_id: Union[str, Path]) -> bool:
    """Check if a Hugging Face model exists locally.

    Args:
        model_id (str or Path):
            Path to the model folder.

    Returns:
        bool:
            Whether the model exists locally.
    """
    try:
        AutoConfig.from_pretrained(model_id)
        return True
    except OSError:
        return False


def get_hf_model_config_locally(model_folder: Union[str, Path]) -> ModelConfig:
    """Get the model configuration from a local Hugging Face model.

    Args:
        model_folder (str or Path):
            Path to the model folder.

    Returns:
        ModelConfig:
            The model configuration.
    """
    # Get the Hugging Face model config, from which we can get the model's label
    # conversions
    config = AutoConfig.from_pretrained(model_folder)

    return ModelConfig(
        model_id=str(model_folder),
        tokenizer_id=str(model_folder),
        revision="main",
        framework=Framework.PYTORCH,
        id2label=config.id2label,
        label2id=config.label2id,
    )
