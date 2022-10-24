"""Main Evaluator class, used to evaluate finetuned models."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .config import EvaluationConfig, TaskConfig
from .enums import CountryCode, Device
from .exceptions import InvalidArchitectureForTask, InvalidEvaluation
from .task_configs import get_all_task_configs
from .task_factory import TaskFactory

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluating finetuned models.

    Args:
        progress_bar (bool, optional):
            Whether progress bars should be shown. Defaults to True.
        save_results (bool, optional):
            Whether to save the benchmark results to
            'aiai_evaluation_results.json'. Defaults to False.
        raise_error_on_invalid_model (bool, optional):
            Whether to raise an error if a model is invalid. Defaults to False.
        cache_dir (str, optional):
            Directory to store cached models. Defaults to '.aiai_cache'.
        use_auth_token (bool or str, optional):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token. Defaults to False.
        track_carbon_emissions (bool, optional):
            Whether to track carbon usage. Defaults to False.
        country_code (CountryCode or str, optional):
            The 3-letter alphabet ISO Code of the country where the compute
            infrastructure is hosted. Only relevant if no internet connection is
            available. Only relevant if `track_carbon_emissions` is set to True.
            Defaults to the empty string. A list of all such codes are available here:
            https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
        prefer_device (Device, optional):
            The device to prefer when evaluating the model. If the device is not
            available then another device will be used. Can be "cuda", "mps" and "cpu".
            Defaults to "cuda".
        only_return_log (bool, optional):
            Whether to only return the log of the evaluation. Defaults to False.
        architecture_fname (str or None, optional):
            The name of the architecture file, if local models are used. If None, the
            architecture file will be automatically detected as the first Python script
            in the model directory. Defaults to None.
        weight_fname (str or None, optional):
            The name of the file containing the model weights, if local models are
            used. If None, the weight file will be automatically detected as the first
            "*.bin" file in the model directory. Defaults to None.
        verbose (bool, optional):
            Whether to output additional output. Defaults to False.

    Attributes:
        evaluation_config (EvaluationConfig):
            The evaluation configuration.
        evaluation_results (dict):
            The evaluation results.
        task_factory (TaskFactory):
            The factory object used to generate tasks to be evaluated.
    """

    def __init__(
        self,
        progress_bar: bool = True,
        save_results: bool = False,
        raise_error_on_invalid_model: bool = False,
        cache_dir: str = ".aiai_cache",
        use_auth_token: Union[bool, str] = False,
        track_carbon_emissions: bool = False,
        country_code: Union[str, CountryCode] = CountryCode.EMPTY,  # type: ignore[attr-defined]
        prefer_device: Device = Device.CUDA,
        only_return_log: bool = False,
        architecture_fname: Optional[str] = None,
        weight_fname: Optional[str] = None,
        verbose: bool = False,
    ):
        # If `country_code` is a string then convert it to a `CountryCode` enum
        if isinstance(country_code, str):
            country_code_enum = CountryCode(country_code.lower())
        else:
            country_code_enum = country_code

        # Build evaluation configuration
        self.evaluation_config = EvaluationConfig(
            raise_error_on_invalid_model=raise_error_on_invalid_model,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            progress_bar=progress_bar,
            save_results=save_results,
            verbose=verbose,
            track_carbon_emissions=track_carbon_emissions,
            country_code=country_code_enum,
            prefer_device=prefer_device,
            architecture_fname=architecture_fname,
            weight_fname=weight_fname,
            only_return_log=only_return_log,
        )

        # Initialise variable storing model lists, so we only have to fetch it once
        self._model_lists: Union[Dict[str, Sequence[str]], None] = None

        # Initialise variable storing all evaluation results, which will be
        # updated as more models are evaluated
        self.evaluation_results: Dict[str, dict] = defaultdict(dict)

        # Set logging level based on verbosity
        logging_level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(logging_level)

        # Initialise a task factory
        self.task_factory = TaskFactory(evaluation_config=self.evaluation_config)

    def evaluate(
        self,
        model_id: Union[Sequence[str], str],
        task: Union[Sequence[str], str],
    ) -> Dict[str, dict]:
        """Evaluates models on datasets.

        Args:
            model_id (str or list of str):
                The model ID(s) of the models to be evaluated.
            task (str or list of str):
                The task(s) to evaluate the model(s) on.

        Returns:
            dict:
                A nested dictionary of the evaluation results. The keys are the names
                of the datasets, with values being new dictionaries having the model
                IDs as keys.
        """
        try:
            # Prepare the model IDs and tasks
            model_ids = self._prepare_model_ids(model_id)
            task_configs = self._prepare_task_configs(task_name=task)

            # If there are multiple models and/or tasks specified, then we log an
            # initial message containing all the upcoming evaluations. The individual
            # (model, task) pairs will also be logged individually later
            if len(model_ids) > 1 or len(task_configs) > 1:

                # Prepare model string for logging
                if len(model_ids) == 1:
                    model_str = f"{model_ids[0]} model"
                else:
                    model_str = ", ".join(model_id for model_id in model_ids[:-1])
                    model_str += f" and {model_ids[-1]} models"

                # Prepare task string for logging
                if len(task_configs) == 1:
                    task_str = f"{task_configs[0].pretty_name} task"
                else:
                    task_str = ", ".join(cfg.pretty_name for cfg in task_configs[:-1])
                    task_str += f" and {task_configs[-1].pretty_name} tasks"

                # Log status
                logger.info(f"Evaluating the {model_str} on the {task_str}.")

            # Evaluate all the models in `model_ids` on all the datasets in
            # `dataset_tasks`
            for task_config in task_configs:
                for m_id in model_ids:
                    try:
                        self._evaluate_single(
                            task_config=task_config,
                            model_id=m_id,
                        )
                    except InvalidArchitectureForTask:
                        logger.info(
                            f"Skipping evaluation of model {m_id} on task "
                            f"{task_config.pretty_name} as the architecture is not "
                            f"compatible with the task."
                        )
                        continue

            # Save the evaluation results
            if self.evaluation_config.save_results:
                output_path = Path.cwd() / "aiai_evaluation_results.json"
                with output_path.open("w") as f:
                    json.dump(self.evaluation_results, f)

            return self.evaluation_results

        except Exception as e:
            if self.evaluation_config.verbose:
                raise e
            else:
                logger.error(f"{type(e).__name__}: {e}")
            return dict(error=dict(type=type(e), message=str(e)))

    def _prepare_model_ids(
        self,
        model_id: Union[Sequence[str], str],
    ) -> List[str]:
        """Prepare the model ID(s) to be evaluated.

        Args:
            model_id (str or list of str):
                The model ID(s) of the models to evaluate.

        Returns:
            sequence of str:
                The prepared list of model IDs.
        """
        model_ids: Sequence[str]
        if isinstance(model_id, str):
            model_ids = [model_id]
        else:
            model_ids = list(model_id)
        return model_ids

    def _prepare_task_configs(
        self,
        task_name: Union[Sequence[str], str],
    ) -> List[TaskConfig]:
        """Prepare the model ID(s) to be evaluated.

        Args:
            task_name (str or list of str):
                The task name(s) to evaluate the model(s) on.

        Returns:
            list of TaskConfig objects:
                The prepared list of task configurations.
        """
        # Create a dictionary that maps evaluation tasks to their associated evaluation
        # task objects
        task_mapping = get_all_task_configs()

        # Create the list of dataset tasks
        if isinstance(task_name, str):
            task_configs = [task_mapping[task_name]]
        else:
            task_configs = [task_mapping[task] for task in task_name]

        return task_configs

    def _evaluate_single(
        self,
        model_id: str,
        task_config: TaskConfig,
    ) -> None:
        """Evaluate a single model on a single task.

        Args:
            model_id (str):
                The model ID to use.
            task_config (TaskConfig):
                The dataset task configuration to use.

        Raises:
            ModelDoesNotExist:
                If the model does not exist on the Hugging Face Hub.
        """
        logger.info(
            f"Evaluating the {model_id} model on the {task_config.pretty_name} task."
        )

        try:
            task = self.task_factory.build_task(task_config)
            results = task(model_id)
            self.evaluation_results[task_config.name][model_id] = results
            logger.debug(f"Results:\n{results}")
        except InvalidEvaluation as e:
            logger.info(
                f"{model_id} could not be evaluated on {task_config.pretty_name}. "
                "Skipping."
            )
            logger.debug(f'The error message was "{e}".')

    def __call__(
        self,
        model_id: Union[Sequence[str], str],
        task: Union[Sequence[str], str],
    ) -> Dict[str, dict]:
        return self.evaluate(model_id=model_id, task=task)
