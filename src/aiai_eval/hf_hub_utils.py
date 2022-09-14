"""Utility functions related to the Hugging Face Hub."""

from typing import Dict, Optional, Union

from huggingface_hub import HfApi, ModelFilter
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils import RepositoryNotFoundError
from requests.exceptions import RequestException
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer

from .config import EvaluationConfig, ModelConfig, TaskConfig
from .enums import Framework
from .exceptions import (
    HuggingFaceHubDown,
    InvalidEvaluation,
    InvalidFramework,
    ModelIsPrivate,
    NoInternetConnection,
)
from .model_adjustment import adjust_model_to_task
from .utils import check_supertask, get_class_by_name, internet_connection_available


def load_model_from_hf_hub(
    model_config: ModelConfig,
    from_flax: bool,
    task_config: TaskConfig,
    evaluation_config: EvaluationConfig,
) -> Dict[str, PreTrainedModel]:
    """Load a PyTorch model.

    Args:
        model_config (ModelConfig):
            The configuration of the model.
        from_flax (bool):
            Whether the model is a Flax model.
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
        InvalidEvaluation:
            If the model either does not have any registered frameworks, or if the
            supertask does not correspond to a Hugging Face AutoModel class.
        ModelIsPrivate:
            If the model is private on the Hugging Face Hub, and `use_auth_token` is
            not set.
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
        model_cls = get_class_by_name(
            class_name=f"auto-model-for-{supertask}",
            module_name="transformers",
        )

        # If the model class could not be found then raise an error
        if not model_cls:
            raise InvalidEvaluation(
                f"The supertask '{supertask}' does not correspond to a Hugging Face "
                " AutoModel type (such as `AutoModelForSequenceClassification`)."
            )

        # Load the model with the correct model class
        model = model_cls.from_pretrained(  # type: ignore[attr-defined]
            model_config.model_id,
            revision=model_config.revision,
            use_auth_token=evaluation_config.use_auth_token,
            config=config,
            cache_dir=evaluation_config.cache_dir,
            from_flax=from_flax,
        )

    # If an error occured then throw an informative exception
    except (OSError, ValueError):

        # If the model is private then raise an informative error
        if model_is_private_on_hf_hub(model_id=model_config.model_id):
            raise ModelIsPrivate(model_id=model_config.model_id)

        # Otherwise, the model does not have any frameworks registered, so raise an
        # error
        else:
            raise InvalidEvaluation(
                f"The model {model_config.model_id} does not have any frameworks "
                "registered."
            )

    # Ensure that the model is compatible with the task
    adjust_model_to_task(
        model=model,
        model_config=model_config,
        task_config=task_config,
    )

    # If the model is a subclass of a RoBERTa model then we have to add a prefix space
    # to the tokens, by the way the model is constructed.
    m_id = model_config.model_id
    prefix = "Roberta" in type(model).__name__
    params = dict(use_fast=True, add_prefix_space=prefix)
    tokenizer = AutoTokenizer.from_pretrained(
        m_id,
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

    return dict(model=model, tokenizer=tokenizer)


def get_hf_hub_model_info(model_id: str) -> ModelInfo:
    """Fetches information about a model on the Hugging Face Hub.

    Args:
        model_id (str):
            The model ID to check.

    Returns:
        ModelInfo:
            The model information.

    Raises:
        RepositoryNotFoundError:
            If the model does not exist on the Hugging Face Hub.
        HuggingFaceHubDown:
            If the model id exists, we are able to request other adresses,
            but we failed to fetch the desired model.
        NoInternetConnection:
            We are not able to request other adresses.
    """
    # Extract the revision from the model_id, if present
    model_id, revision = model_id.split("@") if "@" in model_id else (model_id, "main")

    # Connect to the Hugging Face Hub API
    hf_api = HfApi()

    # Get the model info, and return it
    try:
        return hf_api.model_info(repo_id=model_id, revision=revision)

    # If the repository was not found on Hugging Face Hub then raise that error
    except RepositoryNotFoundError as e:
        raise e

    # If fetching from the Hugging Face Hub failed in a different way then throw a
    # reasonable exception
    except RequestException:
        if internet_connection_available():
            raise HuggingFaceHubDown()
        else:
            raise NoInternetConnection()


def model_is_private_on_hf_hub(model_id: str) -> Union[bool, None]:
    """Checkes whether a model is private on the Hugging Face Hub.

    Args:
        model_id (str):
            The model ID to check.

    Returns:
        bool or None:
            If model is private on the Hugging Face Hub or not. If model does not exist
            on the Hub at all then it returns None.
    """
    try:
        model_info = get_hf_hub_model_info(model_id=model_id)
        return model_info.private
    except RepositoryNotFoundError:
        return None


def model_exists_on_hf_hub(model_id: str) -> Union[bool, None]:
    """Checks whether a model exists on the Hugging Face Hub.

    Args:
        model_id (str):
            The model ID to check.

    Returns:
        bool:
            If model exists on the Hugging Face Hub or not.
    """
    try:
        get_hf_hub_model_info(model_id=model_id)
        return True
    except RepositoryNotFoundError:
        return False


def get_model_config_from_hf_hub(
    model_id: str,
    evaluation_config: EvaluationConfig,
) -> ModelConfig:
    """Function to get the model configuration from the Hugging Face Hub.

    Args:
        model_id (str):
            The Hugging Face ID of the model.
        evaluation_config (EvaluationConfig):
            The evaluation configuration.

    Returns:
        ModelConfig:
            The model configuration.

    Raises:
        HuggingFaceHubDown:
            If the model id exists, we are able to request other adresses,
            but we failed to fetch the desired model.
        NoInternetConnection:
            We are not able to request other adresses.
        InvalidFramework:
            If the model only exists as a TensorFlow model.
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

    # Define the Hugging Face Hub API object
    api = HfApi()

    # Fetch the model metadata from the Hugging Face Hub
    try:
        models = api.list_models(
            filter=ModelFilter(author=author, model_name=model_name),
            use_auth_token=evaluation_config.use_auth_token,
        )

    # If fetching from the Hugging Face Hub failed then throw a reasonable exception
    except RequestException:
        if internet_connection_available():
            raise HuggingFaceHubDown()
        else:
            raise NoInternetConnection()

    # Filter the models to only keep the one with the specified model ID
    models = [model for model in models if model.modelId == model_id_without_revision]

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