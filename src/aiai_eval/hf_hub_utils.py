"""Utility functions related to the Hugging Face Hub."""

from typing import Dict, List, Optional, Tuple, Union

from huggingface_hub import HfApi, ModelFilter
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils import RepositoryNotFoundError
from requests.exceptions import RequestException
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.processing_auto import AutoProcessor
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
            tokenizer or processor.

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
        allowed_architectures = (
            task_config.architectures if task_config.architectures else []
        )
        (
            supertask_which_is_valid_architecture,
            allowed_and_checked_architectures,
        ) = check_supertask(
            architectures=config.architectures,
            supertask=supertask,
            allowed_architectures=allowed_architectures,
        )

        # Get the model class associated with the supertask
        if supertask_which_is_valid_architecture:
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
        else:
            raise InvalidEvaluation(
                f"Could not find a valid architecture for the model {model_config.model_id}."
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
        private_model = model_is_private_on_hf_hub(
            model_id=model_config.model_id,
            use_auth_token=evaluation_config.use_auth_token,
        )
        if private_model:
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
    tokenizer_id = model_config.tokenizer_id
    prefix = "Roberta" in type(model).__name__
    params = dict(use_fast=True, add_prefix_space=prefix)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=model_config.revision,
        use_auth_token=evaluation_config.use_auth_token,
        **params,
    )

    # Try to load a processor from the model id, if it does not exist, then set
    # processor to None
    try:
        processor_id = model_config.processor_id
        processor = AutoProcessor.from_pretrained(
            processor_id,
            revision=model_config.revision,
            use_auth_token=evaluation_config.use_auth_token,
        )
    except OSError:
        processor = None

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

    return dict(model=model, tokenizer=tokenizer, processor=processor)


def get_hf_hub_model_info(
    model_id: str,
    use_auth_token: Union[bool, str],
) -> ModelInfo:
    """Fetches information about a model on the Hugging Face Hub.

    Args:
        model_id (str):
            The model ID to check.
        use_auth_token (bool or str):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token.

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
        token = None if isinstance(use_auth_token, bool) else use_auth_token
        return hf_api.model_info(
            repo_id=model_id,
            revision=revision,
            token=token,
        )

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


def model_is_private_on_hf_hub(
    model_id: str,
    use_auth_token: Union[bool, str],
) -> Union[bool, None]:
    """Checkes whether a model is private on the Hugging Face Hub.

    Args:
        model_id (str):
            The model ID to check.
        use_auth_token (bool or str):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token.

    Returns:
        bool or None:
            If model is private on the Hugging Face Hub or not. If model does not exist
            on the Hub at all then it returns None.
    """
    try:
        model_info = get_hf_hub_model_info(
            model_id=model_id, use_auth_token=use_auth_token
        )
        return model_info.private
    except RepositoryNotFoundError:
        return None


def model_exists_on_hf_hub(
    model_id: str,
    use_auth_token: Union[bool, str],
) -> Union[bool, None]:
    """Checks whether a model exists on the Hugging Face Hub.

    Args:
        model_id (str):
            The model ID to check.
        use_auth_token (bool or str):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token.

    Returns:
        bool:
            If model exists on the Hugging Face Hub or not.
    """
    try:
        get_hf_hub_model_info(model_id=model_id, use_auth_token=use_auth_token)
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

    # Extract the model ID
    model_id = models[0].modelId

    # Get the label conversions
    id2label, label2id = get_label_conversions(
        model_id=model_id,
        revision=revision,
        use_auth_token=evaluation_config.use_auth_token,
    )

    # Construct and return the model config
    return ModelConfig(
        model_id=model_id,
        tokenizer_id=model_id,
        processor_id=model_id,
        framework=framework,
        revision=revision,
        id2label=id2label,
        label2id=label2id,
    )


def get_label_conversions(
    model_id: str,
    revision: str,
    use_auth_token: Union[bool, str],
) -> Tuple[Union[List[str], None], Union[Dict[str, int], None]]:
    """Function to get the label conversions from the Hugging Face Hub.

    Args:
        model_id (str):
            The Hugging Face ID of the model.
        revision (str):
            The revision of the model.
        use_auth_token (bool or str):
            Whether to use the authentication token or not, or the token itself.

    Returns:
        pair of list and dict:
            The `id2label` mapping and the `label2id` mapping. Either of them can be
            None, if the model does not have a label mapping.
    """
    # Attempt to fetch the model config from the Hugging Face Hub, if it exists
    try:

        # Download the model config from the Hugging Face Hub
        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            use_auth_token=use_auth_token,
        )

        # Extract the `id2label` conversion from the model config. If it doesn't exist
        # then we set it to None
        try:
            id2label = config.id2label
        except AttributeError:
            id2label = None

        # Ensure that the `id2label` conversion is a list
        if isinstance(id2label, dict):
            try:
                id2label = [id2label[idx] for idx in range(len(id2label))]
            except KeyError:
                raise InvalidEvaluation(
                    "There is a gap in the indexing dictionary of the model."
                )

        # Make all labels upper case
        if id2label is not None:
            id2label = [label.upper() for label in id2label]

        # Create `label2id` conversion from `id2label`
        if id2label:
            label2id = {label: idx for idx, label in enumerate(id2label)}
        else:
            label2id = None

    # If the model config does not exist on the Hugging Face Hub, then we set the label
    # conversions to None
    except OSError:
        id2label = None
        label2id = None

    # Return the label conversions
    return id2label, label2id
