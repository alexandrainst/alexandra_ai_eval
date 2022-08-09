"""Main Evaluator class, used to evaluate models."""


import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .config import DatasetTask, EvaluationConfig
from .exceptions import InvalidEvaluation
from .hf_hub import get_model_lists
from .task_configs import get_all_dataset_tasks
from .task_factory import TaskFactory

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluating provided Danish language models.

    Args:
        progress_bar (bool, optional):
            Whether progress bars should be shown. Defaults to True.
        save_results (bool, optional):
            Whether to save the benchmark results to
            'aiai_eval_results.json'. Defaults to False.
        model_task (str or sequence of str, optional):
            The tasks to include for models. If "all" then models will not be filtered
            based on the task they were trained on. Defaults to "all".
        dataset_task (str or sequence of str, optional):
            The tasks to include for dataset. If "all" then datasets will not be
            filtered based on their task. Defaults to "all".
        raise_error_on_invalid_model (bool, optional):
            Whether to raise an error if a model is invalid. Defaults to False.
        cache_dir (str, optional):
            Directory to store cached models. Defaults to '.aiai_eval_cache'.
        use_auth_token (bool or str, optional):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token. Defaults to False.
        verbose (bool, optional):
            Whether to output additional output. Defaults to False.

    Attributes:
        progress_bar (bool): Whether progress bars should be shown.
        save_results (bool): Whether to save the benchmark results.
        model_task (str or list of str): The model tasks to include.
        dataset_task (str or list of str): The dataset tasks to include.
        verbose (bool): Whether to output additional output.
        use_auth_token (str or bool): The authentication token for the Hugging Face Hub.
        evaluation_results (dict): The benchmark results.
    """

    def __init__(
        self,
        progress_bar: bool = True,
        save_results: bool = False,
        model_task: Optional[Union[str, Sequence[str]]] = None,
        dataset_task: Optional[Union[str, Sequence[str]]] = None,
        raise_error_on_invalid_model: bool = False,
        cache_dir: str = ".aiai_eval_cache",
        use_auth_token: Union[bool, str] = False,
        verbose: bool = False,
    ):
        # Build dataset tasks
        dataset_tasks = self._prepare_dataset_tasks(dataset_task)

        # Build evaluation configuration
        self.evaluation_config = EvaluationConfig(
            model_tasks=model_task,
            dataset_tasks=dataset_tasks,
            raise_error_on_invalid_model=raise_error_on_invalid_model,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            progress_bar=progress_bar,
            save_results=save_results,
            verbose=verbose,
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
    ) -> Dict[str, Dict[str, dict]]:
        """Evaluates models on datasets.

        Args:
            model_id (str or list of str):
                The model ID(s) of the models to be evaluated.
            dataset (str or list of str):
                The dataset(s) to evaluate the model(s) on.

        Returns:
            dict:
                A nested dictionary of the evaluation results. The keys are the names of
                the datasets, with values being new dictionaries having the model IDs
                as keys.
        """
        # Prepare the model IDs
        model_ids = self._prepare_model_ids(model_id)

        # Evaluate all the models in `model_ids` on all the datasets in `dataset_tasks`
        dataset_tasks = self.evaluation_config.dataset_tasks
        for dataset_task in dataset_tasks:
            for m_id in model_ids:
                self._evaluate_single(
                    dataset_task=dataset_task,
                    model_id=m_id,
                )

        # Save the evaluation results
        if self.evaluation_config.save_results:
            output_path = Path.cwd() / "aiai_eval_results.json"
            with output_path.open("w") as f:
                json.dump(self.evaluation_results, f)

        return self.evaluation_results

    def _prepare_model_ids(
        self,
        model_id: Optional[Union[Sequence[str], str]],
    ) -> Sequence[str]:
        """Prepare the model ID(s) to be evaluated.
        Args:
            model_id (str, list of str or None):
                The model ID(s) of the models to evaluate. If None then all model IDs
                will be retrieved.
        Returns:
            sequence of str:
                The prepared list of model IDs.
        """
        model_ids: Sequence[str]
        if model_id is None:
            model_ids = self._get_fresh_model_ids(
                tasks=self.evaluation_config.model_tasks,
            )
        elif isinstance(model_id, str):
            model_ids = [model_id]
        else:
            model_ids = model_id

        return model_ids

    def _evaluate_single(
        self,
        dataset_task: DatasetTask,
        model_id: str,
    ):
        """Evaluate a single model on a single task.
        Args:
            dataset_task (DatasetTask):
                The dataset task configuration to use.
            model_id (str):
                The model ID to use.
        """
        logger.info(f"Evaluating {model_id} on {dataset_task.pretty_dataset_name}")
        try:
            # dataset_obj = self.task_factory.build_dataset(dataset_task)
            results = (
                "NA"  # dataset_obj(model_id) # NB NEED TO IMPLEMENT `build_dataset`
            )
            self.evaluation_results[dataset_task.name][model_id] = results
            logger.debug(f"Results:\n{results}")

        except InvalidEvaluation as e:

            # If the model ID is not valid then raise an error, if specified
            model_err_msg = "does not exist on the Hugging Face Hub"
            if (
                self.evaluation_config.raise_error_on_invalid_model
                and model_err_msg in str(e)
            ):
                raise e

            # Otherwise, log the error
            else:
                logger.info(
                    f"{model_id} could not be evaluated on "
                    f"{dataset_task.pretty_dataset_name}. Skipping."
                )
                logger.debug(f'The error message was "{e}".')

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def _get_fresh_model_ids(
        self,
        tasks: Optional[Sequence[str]],
    ) -> list:
        """Get list of model IDs from the Hugging Face Hub.

        Args:
            tasks (None or sequence of str):
                The tasks of the models to fetch. If None then the models will not be
                filtered on tasks.

        Returns:
            list:
                List of model IDs.
        """
        # Specify boolean variables determining whether the input variables are new
        new_tasks = (
            self._model_lists is not None
            and tasks is not None
            and any(task not in self._model_lists for task in tasks)
        )

        # If the model lists have not been fetched already, then do it
        if self._model_lists is None or new_tasks:
            self._model_lists = get_model_lists(
                tasks=tasks,
                use_auth_token=self.evaluation_config.use_auth_token,
            )

        # Extract all the model IDs from the model lists
        model_ids: List[str] = list()
        if tasks is not None:
            for task in tasks:
                model_ids.extend(self._model_lists[task])  # type: ignore
        model_ids.extend(self._model_lists["multilingual"])  # type: ignore

        # Remove duplicate model IDs
        model_ids = list(set(model_ids))

        return model_ids

    def _prepare_dataset_tasks(
        self, dataset_task: Optional[Union[str, Sequence[str]]]
    ) -> Sequence[DatasetTask]:
        """Prepare dataset task(s) for evaluation.
        Args:
            dataset_task (str or sequence of str, optional):
                The tasks to include for dataset. If "all" then datasets will not be
                filtered based on their task. Defaults to "all".

        Returns:
            sequence of DatasetTask:
                The prepared dataset tasks.
        """
        # Create a dictionary that maps evaluation tasks to their associated evaluation
        # task objects
        dataset_task_mapping = get_all_dataset_tasks()

        # Create the list of dataset tasks
        if dataset_task is None:
            dataset_tasks = list(dataset_task_mapping.values())
        elif isinstance(dataset_task, str):
            dataset_tasks = [dataset_task_mapping[dataset_task]]
        else:
            dataset_tasks = [dataset_task_mapping[task] for task in dataset_task]

        return dataset_tasks

    def _prepare_model_tasks(
        self, model_task: Optional[Union[str, Sequence[str]]]
    ) -> Optional[Sequence[str]]:
        """Prepare model task(s) for evaluation.
        Args:
            model_task (str or list of str):
                The tasks to include for models. If "all" then models will not be
                filtered based on the task they were trained on.

        Returns:
            None or sequence of str:
                The prepared model tasks.
        """
        return [model_task] if isinstance(model_task, str) else model_task
