"""Main Evaluator class, used to evaluate finetuned models."""

import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from requests.exceptions import ConnectionError, RequestException
from tabulate import tabulate

from .config import EvaluationConfig, TaskConfig
from .enums import CountryCode, Device
from .exceptions import InvalidArchitectureForTask, InvalidEvaluation
from .leaderboard_utils import Session
from .task_configs import get_all_task_configs
from .task_factory import TaskFactory

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluating finetuned models.

    Args:
        progress_bar:
            Whether progress bars should be shown. Defaults to True.
        save_results:
            Whether to save the benchmark results to
            'alexandra_ai_evaluation_results.json'. Defaults to False.
        send_results_to_leaderboard:
            Whether to send the benchmark results to the leaderboard. Defaults to
            True.
        leaderboard_url:
            The URL of the leaderboard. Defaults to
            'https://api.aiai.alexandrainst.dk'.
        raise_error_on_invalid_model:
            Whether to raise an error if a model is invalid. Defaults to False.
        cache_dir:
            Directory to store cached models. Defaults to '.alexandra_ai_cache'.
        token:
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token. Defaults to False.
        track_carbon_emissions:
            Whether to track carbon usage. Defaults to False.
        country_code:
            The 3-letter alphabet ISO Code of the country where the compute
            infrastructure is hosted. Only relevant if no internet connection is
            available. Only relevant if `track_carbon_emissions` is set to True.
            Defaults to the empty string. A list of all such codes are available here:
            https://en.wikipedia.org/wiki/list_of_ISO_3166_country_codes
        prefer_device:
            The device to prefer when evaluating the model. If the device is not
            available then another device will be used. Can be "cuda", "mps" and "cpu".
            Defaults to "cuda".
        only_return_log:
            Whether to only return the log of the evaluation. Defaults to False.
        architecture_fname:
            The name of the architecture file, if local models are used. If None, the
            architecture file will be automatically detected as the first Python script
            in the model directory. Defaults to None.
        weight_fname:
            The name of the file containing the model weights, if local models are
            used. If None, the weight file will be automatically detected as the first
            "*.bin" file in the model directory. Defaults to None.
        verbose:
            Whether to output additional output. Defaults to False.

    Attributes:
        evaluation_config:
            The evaluation configuration.
        evaluation_results:
            The evaluation results.
        task_factory:
            The factory object used to generate tasks to be evaluated.
    """

    def __init__(
        self,
        progress_bar: bool = True,
        save_results: bool = False,
        send_results_to_leaderboard: bool = True,
        leaderboard_url: str = "https://api.aiai.alexandrainst.dk",
        raise_error_on_invalid_model: bool = False,
        cache_dir: str = ".alexandra_ai_cache",
        token: bool | str = False,
        track_carbon_emissions: bool = False,
        country_code: str
        | CountryCode = (CountryCode.EMPTY),  # type: ignore[attr-defined]
        prefer_device: Device = Device.CUDA,
        only_return_log: bool = False,
        architecture_fname: str | None = None,
        weight_fname: str | None = None,
        verbose: bool = False,
    ):
        if isinstance(country_code, str):
            country_code_enum = CountryCode(country_code.lower())
        else:
            country_code_enum = country_code

        self.evaluation_config = EvaluationConfig(
            raise_error_on_invalid_model=raise_error_on_invalid_model,
            cache_dir=cache_dir,
            token=token,
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
        self._model_lists: dict[str, list[str]] | None = None

        # Initialise variable storing all evaluation results, which will be
        # updated as more models are evaluated
        self.evaluation_results: dict[str, dict] = defaultdict(dict)

        logging_level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(logging_level)

        self.task_factory = TaskFactory(evaluation_config=self.evaluation_config)

        self.send_results_to_leaderboard = send_results_to_leaderboard
        self.leaderboard_url = leaderboard_url
        self.leaderboard_client = (
            Session(
                base_url=self.leaderboard_url,
            )
            if self.send_results_to_leaderboard
            else None
        )

    def evaluate(
        self,
        model_id: list[str] | str,
        task: list[str] | str,
    ) -> dict[str, dict]:
        """Evaluates models on datasets.

        Args:
            model_id:
                The model ID(s) of the models to be evaluated.
            task:
                The task(s) to evaluate the model(s) on.

        Returns:
            A nested dictionary of the evaluation results. The keys are the names of
            the datasets, with values being new dictionaries having the model IDs as
            keys.

        Raises:
            RequestException:
                If the request to the leaderboard fails.
        """
        try:
            model_ids = self._prepare_model_ids(model_id)
            task_configs = self._prepare_task_configs(task_name=task)

            # If there are multiple models and/or tasks specified, then we log an
            # initial message containing all the upcoming evaluations. The individual
            # (model, task) pairs will also be logged individually later
            if len(model_ids) > 1 or len(task_configs) > 1:
                if len(model_ids) == 1:
                    model_str = f"{model_ids[0]} model"
                else:
                    model_str = ", ".join(model_id for model_id in model_ids[:-1])
                    model_str += f" and {model_ids[-1]} models"

                if len(task_configs) == 1:
                    task_str = f"{task_configs[0].pretty_name} task"
                else:
                    task_str = ", ".join(cfg.pretty_name for cfg in task_configs[:-1])
                    task_str += f" and {task_configs[-1].pretty_name} tasks"

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
                output_path = Path.cwd() / "alexandra_ai_evaluation_results.json"
                with output_path.open("w") as f:
                    json.dump(self.evaluation_results, f)

            # Send the evaluation results to the leaderboard
            if self.send_results_to_leaderboard:
                if all(self._send_results_to_leaderboard()):
                    logger.info("Successfully sent results to leaderboard.")
                else:
                    raise RequestException("Failed to send result(s) to leaderboard.")

            return self.evaluation_results

        except Exception as e:
            if self.evaluation_config.verbose:
                raise e
            else:
                logger.error(f"{type(e).__name__}: {e}")
            return dict(error=dict(type=type(e), message=str(e)))

    def _prepare_model_ids(
        self,
        model_id: list[str] | str,
    ) -> list[str]:
        """Prepare the model ID(s) to be evaluated.

        Args:
            model_id:
                The model ID(s) of the models to evaluate.

        Returns:
            The prepared list of model IDs.
        """
        model_ids: list[str]
        if isinstance(model_id, str):
            model_ids = [model_id]
        else:
            model_ids = list(model_id)
        return model_ids

    def _prepare_task_configs(
        self,
        task_name: list[str] | str,
    ) -> list[TaskConfig]:
        """Prepare the model ID(s) to be evaluated.

        Args:
            task_name:
                The task name(s) to evaluate the model(s) on.

        Returns:
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

        # Set task config architecture field to supertask if it is not already set
        for task_config in task_configs:
            if task_config.architectures is None:
                task_config.architectures = [task_config.supertask]

        return task_configs

    def _evaluate_single(
        self,
        model_id: str,
        task_config: TaskConfig,
    ) -> None:
        """Evaluate a single model on a single task.

        Args:
            model_id:
                The model ID to use.
            task_config:
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

    def _send_results_to_leaderboard(self) -> list[bool]:
        """Send the evaluation results to the leaderboard.

        Args:
            results (dict):
                The evaluation results to send.

        Returns:
            A list of booleans indicating whether the results were successfully sent to
            the leaderboard.
        """
        error_if_client_is_none = (
            "Leaderboard client is not initialized. Ensure that "
            "`self.send_results_to_leaderboard is set to True and try re-initialising "
            "the evaluator."
        )
        assert self.leaderboard_client is not None, error_if_client_is_none

        self.leaderboard_client.check_connection()

        status: list[bool] = list()

        # Loop through the evaluation results and send each one to the leaderboard
        for task_name in self.evaluation_results.keys():
            for model_id in self.evaluation_results[task_name].keys():
                logger.info(
                    f"Sending results for {model_id} to the {task_name}-leaderboard."
                )
                # Post the results to the leaderboard and log the response
                try:
                    task_leaderboard_json = self.leaderboard_client.post_model_to_task(
                        model_type=self.evaluation_results[task_name][model_id][
                            "model_type"
                        ],
                        task_name=task_name,
                        model_id=model_id,
                        metrics=self.evaluation_results[task_name][model_id]["total"],
                        test=self.evaluation_config.testing,
                    )
                    # Check if response contains an error message
                    if "error" in task_leaderboard_json:
                        error_msg = task_leaderboard_json["error"]
                        logger.info(
                            f"Could not send results for {model_id} to the "
                            f"{task_name}-leaderboard. Skipping."
                        )
                        logger.debug(f'The error message was "{error_msg}".')

                        # Append the status of the leaderboard post to the status list
                        status.append(False)
                        continue

                    logger.info(
                        f"Results successfully sent to the {task_name}-leaderboard."
                    )
                except (ValueError, ConnectionError) as e:
                    logger.info(
                        f"Could not send results for {model_id} to the "
                        f"{task_name}-leaderboard. Skipping."
                    )
                    logger.debug(f'The error message was "{e}".')

                    # Append the status of the leaderboard post to the status list
                    status.append(False)
                    continue

                # Make dataframe from leaderboard json
                task_leaderboard = pd.DataFrame.from_dict(task_leaderboard_json)

                # Get metric columns, i.e. all columns except "id", "model_type" and
                # "model_id"
                metric_columns = task_leaderboard.columns.difference(
                    ["id", "model_type", "model_id"]
                )

                # Sort the leaderboard by the average of the metric columns
                task_leaderboard["average_score"] = (
                    task_leaderboard[metric_columns].dropna(axis=1).mean(axis=1)
                )
                task_leaderboard.sort_values("average_score")

                # Log the leaderboard
                tabulated_leaderboard = tabulate(
                    tabular_data=task_leaderboard, headers="keys", tablefmt="fancy_grid"
                )
                logger.info(f"Leaderboard:\n{tabulated_leaderboard}")

                # Get the rank of the model, first check it is in the leaderboard and
                # then get the rank. If it is not in the leaderboard, we are running
                # tests.
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                    if model_id in task_leaderboard["model_id"].values:
                        model_rank = (
                            int(
                                task_leaderboard[
                                    task_leaderboard["model_id"] == model_id
                                ].index[0]
                            )
                            + 1
                        )

                        # Log the rank of the model
                        logger.info(
                            f"{model_id} is ranked {model_rank} on the "
                            f"{task_name}-leaderboard."
                        )

                status.append(True)
        return status

    def __call__(
        self,
        model_id: list[str] | str,
        task: list[str] | str,
    ) -> dict[str, dict]:
        return self.evaluate(model_id=model_id, task=task)
