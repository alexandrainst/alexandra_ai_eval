"""Functions related to the Hugging Face Hub."""

from typing import Optional

from huggingface_hub import HfApi, ModelFilter
from huggingface_hub.utils import RepositoryNotFoundError
from requests.exceptions import RequestException

from .config import EvaluationConfig, ModelConfig
from .enums import Framework
from .exceptions import (
    HuggingFaceHubDown,
    InvalidFramework,
    ModelDoesNotExist,
    ModelFetchFailed,
    NoInternetConnection,
)
from .model_loading import load_spacy_model
from .utils import internet_connection_available


def model_exists_on_hf_hub(model_id: str) -> bool:
    """Function checks if `model_id` exists on Huggingface Hub.

    Args:
        model_id (str):
            The model ID to check.

    Returns:
        bool:
            If model exists on Hugginface Hub or not.
    """
    # Extract the revision from the model_id, if present
    model_id, revision = model_id.split("@") if "@" in model_id else (model_id, "main")

    # Connect to the Hugging Face Hub API
    hf_api = HfApi()

    # Check if the model exists
    try:
        hf_api.model_info(repo_id=model_id, revision=revision)
        return True
    except RepositoryNotFoundError:
        return False


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
        load_spacy_model(model_id)
        return True
    except ModelFetchFailed:
        return False


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
        HuggingFaceHubDown:
            If the model id exists, we are able to request other adresses,
            but we failed to fetch the desired model.
        NoInternetConnection:
            We are not able to request other adresses.
    """
    # Extract the revision from the model ID, if it is specified
    if "@" in model_id:
        model_id_without_revision, revision = model_id.split("@", 1)
    else:
        model_id_without_revision = model_id
        revision = "main"

    # Extract the author and model name from the model ID
    author: Optional[str]
    if "/" in model_id_without_revision:
        author, model_name = model_id_without_revision.split("/")
    else:
        author = None
        model_name = model_id_without_revision

    # Check if model exists on Hugging Face Hub or as a spaCy model
    model_on_hf_hub = model_exists_on_hf_hub(model_id=model_id)
    model_on_spacy = model_exists_on_spacy(model_id=model_id)

    # If it does not exist on Hugging Face Hub or as a spaCy model, raise an error
    if not model_on_hf_hub and not model_on_spacy:
        raise ModelDoesNotExist(model_id=model_id)

    # If it exists as a spaCy model, we return the spaCy model config
    if model_on_spacy:
        return ModelConfig(model_id=model_id, revision="", framework=Framework.SPACY)

    # Otherwise it exists on the Hugging Face Hub, and we attempt to fetch the
    # information from there
    try:

        # Define the Hugging Face Hub API object
        api = HfApi()

        # Fetch the model metadata
        models = api.list_models(
            filter=ModelFilter(author=author, model_name=model_name),
            use_auth_token=evaluation_config.use_auth_token,
        )

        # Filter the models to only keep the one with the specified model ID
        models = [
            model for model in models if model.modelId == model_id_without_revision
        ]

        # Fetch the model tags
        tags = models[0].tags

        # Extract the framework, which defaults to PyTorch
        framework = Framework.PYTORCH
        if "pytorch" in tags:
            pass
        elif "jax" in tags:
            framework = Framework.JAX
        elif "spacy" in tags:
            framework = Framework.SPACY
        elif "tf" in tags or "tensorflow" in tags or "keras" in tags:
            raise InvalidFramework("tensorflow")

        # Construct and return the model config
        return ModelConfig(
            model_id=models[0].modelId,
            framework=framework,
            revision=revision,
        )

    # If fetching from the Hugging Face Hub failed then throw a reasonable exception
    except RequestException:

        # Check if it is because the internet is down
        if internet_connection_available():
            raise HuggingFaceHubDown()
        else:
            raise NoInternetConnection()
