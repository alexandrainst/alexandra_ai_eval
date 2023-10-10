"""Utility functions related to loading local PyTorch models."""

import inspect
import json
import logging
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Type, get_type_hints

import torch
import torch.nn as nn
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import EvaluationConfig, ModelConfig, TaskConfig
from .enums import Framework
from .model_adjustment import adjust_model_to_task

logger = logging.getLogger(__name__)


def load_local_pytorch_model(
    model_config: ModelConfig,
    task_config: TaskConfig,
    evaluation_config: EvaluationConfig,
) -> dict[str, nn.Module | PreTrainedTokenizerBase]:
    """Load a local PyTorch model from a path.

    Args:
        model_config:
            The configuration of the model.
        task_config:
            The task configuration.
        evaluation_config:
            The evaluation configuration.

    Returns:
        A dictionary containing the model and tokenizer.

    Raises:
        ValueError:
            If no model architecture file or model weight file is found in the model
            folder, or if the model architecture file does not contain a class
            subclassing `torch.nn.Module`.
    """
    # TEMP: If the model's `id2label` mapping has fewer labels than the dataset, then
    # raise an informative error. This is a temporary fix until we have a better
    # solution for handling this case.
    if model_config.id2label is not None and len(model_config.id2label) < len(
        task_config.id2label
    ):
        raise ValueError(
            f"The model {model_config.model_id!r} has fewer labels than the dataset "
            f"{task_config.name!r} (the model has the labels {model_config.id2label} "
            f"and the dataset has the labels {task_config.id2label}). We do not "
            "currently support this case. If the above-mentioned model labels are "
            "wrong, then please adjust these in the configuration JSON file, located "
            f"in the {model_config.model_id!r} folder."
        )

    # Ensure that the model_folder is a Path object
    model_folder = Path(model_config.model_id)

    # Add the model folder to PATH
    sys.path.insert(0, str(model_folder))

    # If no architecture_fname is provided, then use the first Python script found
    arc_fname = evaluation_config.architecture_fname
    if arc_fname is None:
        try:
            architecture_path = next(model_folder.glob("*.py"))

        # Raise an error if no Python script is found
        except StopIteration:
            raise ValueError(
                f"No Python script found in the model folder {model_folder}."
            )
    else:
        if not arc_fname.endswith(".py"):
            arc_fname = arc_fname + ".py"
        architecture_path = model_folder / arc_fname

        # Raise an error if the architecture file does not exist
        if not architecture_path.exists():
            raise ValueError(
                f"The model architecture file {architecture_path} does not exist."
            )

    # If no weight_fname is provided, then use the first file found ending with ".bin"
    if evaluation_config.weight_fname is None:
        try:
            weight_path = next(model_folder.glob("*.bin"))

        # Raise an error if no weight file is found
        except StopIteration:
            raise ValueError(
                f"No model weights found in the model folder {model_folder}."
            )
    else:
        weight_path = model_folder / evaluation_config.weight_fname

        # Raise an error if the weight file does not exist
        if not weight_path.exists():
            raise ValueError(f"The model weights file {weight_path} does not exist.")

    # Import the module containing the model architecture
    module_name = architecture_path.stem
    module = import_module(module_name)

    # Get the candidates for the model architecture class, being all classes in the
    # loaded module that are subclasses of `torch.nn.Module`, and which come from the
    # desired module (as opposed to being imported from elsewhere)
    model_candidates = [
        obj
        for _, obj in module.__dict__.items()
        if isinstance(obj, type)
        and issubclass(obj, nn.Module)
        and obj.__module__ == module_name
    ]

    # If there are no candidates, raise an error
    if not model_candidates:
        raise ValueError(f"No model architecture found in {architecture_path}")

    # Pick the first candidate
    model_cls = model_candidates[0]

    # Get the arguments for the class initializer
    model_args = list(inspect.signature(model_cls).parameters)

    # Remove the arguments that have default values
    defaults_tuple = model_cls.__init__.__defaults__  # type: ignore[misc]
    if defaults_tuple is not None:
        model_args = model_args[: -len(defaults_tuple)]

    # Get the type hints for the class initializer
    type_hints = get_type_hints(model_cls.__init__)  # type: ignore[misc]

    # If any of the arguments are not in the type hints, raise an error
    for arg in model_args:
        if arg not in type_hints:
            raise ValueError(
                f"A type hint or default value for the {arg!r} argument of the "
                f"{model_cls.__name__!r} class is missing. Please specify either "
                f"in the {architecture_path.name!r} file."
            )

    # Fetch the model keyword arguments from the local configuration
    model_kwargs = {
        arg: get_from_config(
            key=arg,
            expected_type=type_hints[arg],
            model_folder=model_folder,
        )
        for arg in model_args
    }

    # Initialize the model with the (potentially empty) set of keyword arguments
    model = model_cls(**model_kwargs)

    # Load the model weights
    state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    model.eval()
    model.to(evaluation_config.device)

    adjust_model_to_task(
        model=model,
        model_config=model_config,
        task_config=task_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_id)

    return dict(model=model, tokenizer=tokenizer, model_type="other")


def pytorch_model_exists_locally(
    model_id: str | Path,
    architecture_fname: str | Path | None = None,
    weight_fname: str | Path | None = None,
) -> bool:
    """Check if a PyTorch model exists locally.

    Args:
        model_id:
            Path to the model folder.
        architecture_fname:
            Name of the file containing the model architecture, which is located inside
            the model folder. If None then the first Python script found in the model
            folder will be used. Defaults to None.
        weight_fname:
            Name of the file containing the model weights, which is located inside
            the model folder. If None then the first file found in the model folder
            ending with ".bin" will be used. Defaults to None.

    Returns:
        Whether the model exists locally.
    """
    model_folder = Path(model_id)

    # If no architecture_fname is provided, then use the first Python script found
    if architecture_fname is None:
        try:
            architecture_path = next(model_folder.glob("*.py"))
        except StopIteration:
            return False
    else:
        architecture_path = model_folder / architecture_fname

    # If no weight_fname is provided, then use the first file found ending with ".bin"
    if weight_fname is None:
        try:
            weight_path = next(model_folder.glob("*.bin"))
        except StopIteration:
            return False
    else:
        weight_path = model_folder / weight_fname

    # Check if the model architecture and weights exist
    return architecture_path.exists() and weight_path.exists()


def get_pytorch_model_config_locally(
    model_folder: str | Path,
    dataset_id2label: list[str],
) -> ModelConfig:
    """Get the model configuration from a local PyTorch model.

    Args:
        model_folder:
            Path to the model folder.
        dataset_id2label:
            list of labels in the dataset.

    Returns:
        The model configuration.
    """
    return ModelConfig(
        model_id=Path(model_folder).name,
        tokenizer_id=get_from_config(
            key="tokenizer_id",
            expected_type=str,
            model_folder=model_folder,
            user_prompt="Please specify the Hugging Face ID of the tokenizer to use: ",
        ),
        processor_id=get_from_config(
            key="processor_id",
            expected_type=str,
            model_folder=model_folder,
            user_prompt="Please specify the Hugging Face ID of the processor to use: ",
        ),
        revision="",
        framework=Framework.PYTORCH,
        id2label=get_from_config(
            key="id2label",
            expected_type=list,
            model_folder=model_folder,
            user_prompt="Please specify the labels in the order the model was trained "
            "(comma-separated), or press enter to use the default values "
            f"[{', '.join(dataset_id2label)}]: ",
            user_prompt_default_value=dataset_id2label,
        ),
    )


def get_from_config(
    key: str,
    expected_type: Type,
    model_folder: str | Path,
    default_value: Any | None = None,
    user_prompt: str | None = None,
    user_prompt_default_value: Any | None = None,
) -> Any:
    """Get an attribute from the local model configuration.

    If the attribute is not found in the local model configuration, then the user
    will be prompted to enter it, after which it will be saved to the local model
    configuration. If the configuration file does not exist, then a new one will be
    created named `config.json`.

    Args:
        key:
            The key to get from the configuration.
        expected_type:
            The expected type of the value.
        model_folder:
            Path to the model folder.
        default_value:
            The default value to use if the attribute is not found in the local model
            configuration. If None then the user will be prompted to enter the value.
            Defaults to None.
        user_prompt:
            The prompt to show the user when asking for the value. If None then the
            prompt will be automatically generated. Defaults to None.
        user_prompt_default_value:
            The default value that a user can press Enter to use, when prompted. If
            None then the user cannot choose a default value. Defaults to None.

    Returns:
        The value of the key, of data type `expected_type`.
    """
    model_folder = Path(model_folder)

    # Get the candidate configuration files
    config_paths = list(model_folder.glob("*.json"))

    # If there isn't a config then we set it to a blank dictionary. Otherwise, we load
    # the config file
    if not config_paths:
        config_path = model_folder / "config.json"
        config = dict()
    else:
        config_path = config_paths[0]
        config = json.loads(config_path.read_text())

    # If the key is not in the config then we either use the default value or prompt
    # the user to enter it
    if key not in config:
        # If the default value is set and is of the correct type, then we use it
        if default_value is not None and isinstance(default_value, expected_type):
            config[key] = default_value

        # Otherwise, we prompt the user to enter the value
        else:
            if user_prompt is None:
                base_prompt = (
                    f"The configuration did not contain the {key!r} entry. Please "
                    "specify its value"
                )
                if user_prompt_default_value is not None:
                    user_prompt = (
                        f"The configuration did not contain the {key!r} entry. Press "
                        f"Enter to use the default value {user_prompt_default_value!r} "
                        "or specify a new value"
                    )

                if expected_type is bool:
                    user_prompt = f"{base_prompt} (true/false): "
                elif expected_type is list:
                    user_prompt = f"{base_prompt} (comma-separated): "
                elif expected_type is dict:
                    user_prompt = f"{base_prompt} (key=value, comma-separated): "
                else:
                    user_prompt = f"{base_prompt}: "

            # Prompt the user to enter the value
            config[key] = get_missing_key_value_from_user(
                user_prompt=user_prompt,
                expected_type=expected_type,
                default_value=user_prompt_default_value,
            )

        # Save the new modified config
        if not config_path.exists():
            config_path.touch()
        config_path.write_text(json.dumps(config, indent=4))

    return config[key]


def get_missing_key_value_from_user(
    user_prompt: str | None,
    expected_type: type,
    default_value: Any | None = None,
):
    """Get a missing key from the user.

    Args:
        user_prompt:
            The prompt to show the user when asking for the value. If None then the
            prompt will be automatically generated.
        expected_type:
            The expected type of the value.
        default_value:
            The default value that a user can press Enter to use, when prompted. If
            None then the user cannot choose a default value. Defaults to None.

    Returns:
        The value of the key, of data type `expected_type`.
    """
    user_input = input(user_prompt)

    # If the user input is blank (i.e. they pressed Enter) and there is a default
    # value, then we use the default value
    if not user_input and default_value is not None:
        return default_value

    # Otherwise, we parse the user input, depending on the expected type
    else:
        # We try to parse the input, and if it fails then we prompt the user to enter
        # it again
        while True:
            try:
                if expected_type is int:
                    return int(user_input)
                elif expected_type is float:
                    return float(user_input)
                elif expected_type is bool:
                    return user_input.lower() == "true"
                elif expected_type is list:
                    return user_input.split()
                elif expected_type is dict:
                    return dict(item.split("=") for item in user_input.split(","))
                else:
                    return user_input

            except ValueError:
                logger.error(
                    f"The value {user_input!r} is not of type {expected_type.__name__}."
                    " Please try again."
                )
                user_input = input(user_prompt)
