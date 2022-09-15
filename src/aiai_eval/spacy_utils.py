"""Utility functions related to the spaCy library."""

import warnings
from subprocess import CalledProcessError
from typing import Dict

import spacy
from spacy.cli.download import download as download_spacy
from spacy.language import Language

from .config import ModelConfig
from .enums import Framework
from .exceptions import ModelFetchFailed
from .utils import is_module_installed

# Ignore warnings from spaCy. This has to be called after the import, as the
# __init__.py file of spaCy sets the warning levels of spaCy warning W036
warnings.filterwarnings("ignore", module="spacy*")


def load_spacy_model(model_id: str) -> Dict[str, Language]:
    """Load a spaCy model.

    Args:
        model_id (str):
            The ID of the model.

    Returns:
        dict:
            A dictionary containing at least the key 'model', with the value being the
            model.

    Raises:
        ModelFetchFailed:
            If the model could not be downloaded.
    """
    local_model_id = model_id.split("/")[-1]

    # Download the model if it has not already been so
    try:
        if not is_module_installed(local_model_id):
            try:
                download_spacy(model=local_model_id)

            # The download function calls a `sys.exit` at the end of the download, so
            # we catch that here and move on
            except SystemExit:
                pass

    except CalledProcessError as e:
        raise ModelFetchFailed(model_id=local_model_id, error_msg=e.output)

    # Load the model
    try:
        model = spacy.load(local_model_id)
    except OSError as e:
        raise ModelFetchFailed(
            model_id=model_id,
            error_msg=str(e),
            message=(
                f"Download of {model_id} failed, with the following error message: "
                f"{str(e)}."
            ),
        )
    return dict(model=model)


def model_exists_on_spacy(model_id: str) -> bool:
    """Checks if a model exists as a spaCy model.

    Args:
        model_id (str):
            The name of the model.

    Returns:
        bool:
            Whether the model exists as a spaCy model.
    """
    try:
        load_spacy_model(model_id=model_id)
        return True
    except ModelFetchFailed:
        return False


def get_model_config_from_spacy(model_id: str) -> ModelConfig:
    """Get the model configuration from a spaCy model.

    Args:
        model_id (str):
            The ID of the model.

    Returns:
        ModelConfig:
            The model configuration.
    """
    return ModelConfig(
        model_id=model_id,
        tokenizer_id="",
        revision="",
        framework=Framework.SPACY,
        id2label=None,
    )
