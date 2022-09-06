"""Abstract Task class."""

import logging
import random
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .co2 import get_carbon_tracker
from .config import EvaluationConfig, ModelConfig, TaskConfig
from .exceptions import (
    InvalidEvaluation,
    InvalidFramework,
    MPSFallbackNotEnabled,
    PreprocessingFailed,
    UnsupportedModelType,
    WrongFeatureColumnName,
)
from .hf_hub import get_model_config
from .metric_configs import EMISSIONS, POWER
from .model_loading import load_model
from .scoring import log_scores
from .utils import clear_memory, enforce_reproducibility

# Set up a logger
logger = logging.getLogger(__name__)


class Task(ABC):
    """Abstract evaluation task class.

    Args:
        task_config (TaskConfig):
            The configuration of the task.
        evaluation_config (EvaluationConfig):
            The configuration of the evaluation.

    Attributes:
        task_config (TaskConfig):
            The configuration of the task.
        evaluation_config (EvaluationConfig):
            The configuration of the evaluation.
    """

    def __init__(self, task_config: TaskConfig, evaluation_config: EvaluationConfig):
        self.task_config = task_config
        self.evaluation_config = evaluation_config

        # Load the metric functions from the `datasets` library
        self._metrics = {
            metric_cfg.name: load_metric(metric_cfg.huggingface_id)
            for metric_cfg in task_config.metrics
        }

    def evaluate(self, model_id: str) -> Union[Dict[str, Dict[str, float]], str]:
        """Evaluate a model.

        Args:
            model_id (str):
                The full Hugging Face Hub path to the pretrained transformer model. The
                specific model version to use can be added after the suffix '@':
                "model_id@v1.0.0". It can be a branch name, a tag name, or a commit id.

        Returns:
            dict:
                The keys in the dict are 'raw' and 'total', with all the raw scores in
                the first dictionary and the aggregated scores in the second.
        """
        # Fetch the model config
        model_config = get_model_config(
            model_id=model_id, evaluation_config=self.evaluation_config
        )

        # Set random seeds to enforce reproducibility of the randomly initialised
        # weights
        rng = enforce_reproducibility(framework=model_config.framework)

        # Load the model
        model_dict = load_model(
            model_config=model_config,
            task_config=self.task_config,
            evaluation_config=self.evaluation_config,
        )

        # Prepare carbon tracker
        if self.evaluation_config.track_carbon_emissions:
            self.carbon_tracker = get_carbon_tracker(
                task_name=self.task_config.name,
                country_iso_code=self.evaluation_config.country_iso_code,
                verbose=self.evaluation_config.verbose,
            )

        # Load the dataset
        dataset = self._load_data()

        # Remove empty examples from the datasets
        for feat_column in self.task_config.feature_column_names:
            try:
                dataset = dataset.filter(lambda record: len(record[feat_column]) > 0)
            except KeyError:
                raise WrongFeatureColumnName(feat_column)

        # Set variable with number of iterations
        num_iter = 10 if not self.evaluation_config.testing else 2

        if model_config.framework in {"pytorch", "jax"}:
            return self._evaluate_pytorch_jax(
                model_dict=model_dict,
                dataset=dataset,
                rng=rng,
                model_config=model_config,
                num_iter=num_iter,
            )

        elif model_config.framework == "spacy":
            return self._evaluate_spacy(
                model_dict=model_dict,
                dataset=dataset,
                rng=rng,
                model_config=model_config,
                num_iter=num_iter,
            )

        else:
            raise InvalidFramework(model_config.framework)

    def _evaluate_pytorch_jax(
        self,
        model_dict: dict,
        dataset: Dataset,
        rng: np.random.Generator,
        model_config: ModelConfig,
        num_iter: int,
    ) -> Union[Dict[str, Dict[str, float]], str]:
        """Evaluate a PyTorch or JAX model.

        Args:
            model_dict (dict):
                The model dictionary, with keys "model" and "tokenizer".
            dataset (Dataset):
                The test dataset.
            rng (np.random.Generator):
                The random number generator, used to generate bootstrapped versions of
                the test dataset.
            model_config (ModelConfig):
                The model configuration.
            num_iter (int):
                The number of bootstrapped samples of the test dataset to use.

        Returns:
            str or dict:
                If the `only_return_log` is set then a string is returned containing
                the logged evaluation results. Otherwise, a nested dictionary of the
                evaluation results. The keys are the names of the datasets, with values
                being new dictionaries having the model IDs as keys.
        """
        # Extract the model and tokenizer
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]

        # Log the number of parameters in the model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of model parameters: {num_params:,}")

        # If we are testing then truncate the test set
        if self.evaluation_config.testing:
            dataset = dataset.select(range(4))

        # Get bootstrapped datasets
        bootstrapped_datasets = [
            Dataset.from_dict(dataset[rng.integers(0, len(dataset), len(dataset))])
            for _ in range(num_iter)
        ]

        # Preprocess the bootstrapped datasets
        try:
            prepared_datasets = [
                self._preprocess_data(
                    bootstrapped_dataset,
                    framework="pytorch",
                    config=model.config,
                    tokenizer=tokenizer,
                )
                for bootstrapped_dataset in bootstrapped_datasets
            ]

        # If the preprocessing failed then raise an error
        except ValueError:
            raise PreprocessingFailed()

        # Set up progress bar
        if self.evaluation_config.progress_bar:
            itr = tqdm(range(num_iter), desc="Evaluating")
        else:
            itr = range(num_iter)

        # Load the data collator
        data_collator = self._load_data_collator(tokenizer)

        scores = list()
        for idx in itr:
            while True:
                test_itr_scores = self._evaluate_pytorch_jax_single_iteration(
                    idx=idx,
                    model_config=model_config,
                    dataset=bootstrapped_datasets[idx],
                    prepared_dataset=prepared_datasets[idx],
                    data_collator=data_collator,
                )
                # If the iteration was successful then break the while-loop
                if isinstance(test_itr_scores, dict):
                    break

                # Otherwise we encountered an error
                else:
                    raise InvalidEvaluation(
                        "An unknown error occurred during the evaluation of the "
                        f"{idx} iteration. The error message returned was: "
                        f"{str(test_itr_scores)}"
                    )

            scores.append(test_itr_scores)

        # If track_carbon_emissions is true append metrics, to correctly log emissions
        # data. We avoid mutating, so any downstream evaluations will not try to use
        # these.
        metric_configs = list(self.task_config.metrics)
        if self.evaluation_config.track_carbon_emissions:
            metric_configs.append(EMISSIONS)
            metric_configs.append(POWER)

        # Log scores
        all_scores = log_scores(
            task_name=self.task_config.pretty_name,
            metric_configs=metric_configs,
            scores=scores,
            model_id=model_config.model_id,
            only_return_log=self.evaluation_config.only_return_log,
        )
        return all_scores

    def _evaluate_pytorch_jax_single_iteration(
        self,
        idx: int,
        model_config: ModelConfig,
        dataset: Dataset,
        prepared_dataset: Dataset,
        data_collator: DataCollator,
    ) -> Union[dict, Exception]:
        """Run a single iteration of a PyTorch/JAX benchmark.

        Args:
            idx (int):
                The index of the current iteration.
            model_config (ModelConfig):
                The model configuration.
            dataset (Dataset):
                The raw test dataset.
            prepared_dataset (Dataset):
                The preprocessed test dataset.
            data_collator (DataCollator):
                The data collator.

        Returns:
            dict or Exception:
                The keys in the dict correspond to the metrics and values
                the corresponding values.
        """
        try:
            # Set random seeds to enforce reproducibility of the randomly
            # initialised weights
            random.seed(703 + idx)
            np.random.seed(703 + idx)
            torch.manual_seed(703 + idx)
            torch.cuda.manual_seed_all(703 + idx)

            # Reinitialise a new model
            model_dict = load_model(
                model_config=model_config,
                task_config=self.task_config,
                evaluation_config=self.evaluation_config,
            )
            model = model_dict["model"]
            tokenizer = model_dict["tokenizer"]

            # Define batch size, which depends on whether we are testing or not
            batch_size = 2 if self.evaluation_config.testing else 32

            # Create dataloader
            dataloader = DataLoader(
                prepared_dataset,  # type: ignore
                batch_size=batch_size,
                shuffle=True,
                collate_fn=data_collator,
            )

            # Create progress bar
            if self.evaluation_config.progress_bar:
                itr = tqdm(
                    dataloader, desc=f"Evaluating iteration {idx+1}", leave=False
                )
            else:
                itr = dataloader

            # Start carbon emissions tracking
            if self.evaluation_config.track_carbon_emissions:
                self.carbon_tracker.start()

            # Get model predictions
            all_predictions = list()
            with torch.no_grad():
                for batch in itr:

                    # Move the tensors to the correct device
                    batch = {
                        key: value.to(self.evaluation_config.device)
                        for key, value in batch.items()
                    }

                    # If we are dealing with a Hugging Face model then we will use the
                    # entire batch dictionary
                    if isinstance(model, PreTrainedModel):
                        model_predictions = model(**batch).logits

                    # If we are dealing with a PyTorch model, then we will only use the
                    # input_ids
                    elif isinstance(model, nn.Module):
                        model_predictions = model(batch["input_ids"])

                    # Otherwise, we throw an error
                    else:
                        model_type = str(type(model))
                        raise UnsupportedModelType(model_type=model_type)

                    # Move the predictions back to the CPU and convert it to a NumPy
                    # array
                    model_predictions = model_predictions.cpu().numpy()

                    # Collect predictions
                    all_predictions.append(model_predictions)

            # Concatenate all the predictions
            all_predictions = np.concatenate(all_predictions, axis=0)

            # Perform post-processing of predictions
            prepared_predictions_and_labels = self._prepare_predictions_and_labels(
                predictions=all_predictions,
                dataset=dataset,
                prepared_dataset=prepared_dataset,
                id2label=model.config.id2label,  # type: ignore
                cls_token_index=tokenizer.cls_token_id,
            )

            # If there are multiple metrics but only one pair in the
            # `all_predictions_labels` list, we copy our that entry to ensure there is a
            # pair for each metric
            if (
                len(prepared_predictions_and_labels) == 1
                and len(self.task_config.metrics) > 1
            ):
                prepared_predictions_and_labels *= len(self.task_config.metrics)

            # Compute the metrics for each prediction batch
            scores = self._compute_metrics(
                predictions_and_labels=prepared_predictions_and_labels,
            )

            # Stop carbon emissions tracking and store emission metrics
            if self.evaluation_config.track_carbon_emissions:
                self.carbon_tracker.stop()
                emissions_data = self.carbon_tracker.final_emissions_data
                factor = 1_000_000 / len(prepared_dataset)
                scores["carbon_emissions"] = factor * emissions_data.emissions
                scores["energy_consumed"] = factor * emissions_data.energy_consumed

            return scores

        except (RuntimeError, ValueError, IndexError) as e:
            if "PYTORCH_ENABLE_MPS_FALLBACK" in str(e):
                raise MPSFallbackNotEnabled()

            # Prevent memory leaks
            try:
                del model
            except UnboundLocalError:
                pass
            try:
                del model_dict
            except UnboundLocalError:
                pass
            clear_memory()

            # Return the error if it wasn't caught by the above conditionals
            return e

    def _evaluate_spacy(
        self,
        model_dict: dict,
        dataset: Dataset,
        rng: np.random.Generator,
        model_config: ModelConfig,
        num_iter: int,
    ) -> Union[Dict[str, Dict[str, float]], str]:
        """Evaluate a PyTorch or JAX model.

        Args:
            model_dict (dict):
                The model dictionary, with keys "model" and "tokenizer".
            dataset (Dataset):
                The test dataset.
            rng (np.random.Generator):
                The random number generator, used to generate bootstrapped versions of
                the test dataset.
            model_config (ModelConfig):
                The model configuration.
            num_iter (int):
                The number of bootstrapped samples of the test dataset to use.

        Returns:
            dict:
                The keys in the dict are 'raw' and 'total', with all the raw scores in
                the first dictionary and the aggregated scores in the second.
        """
        raise NotImplementedError

    def _compute_metrics(
        self,
        predictions_and_labels: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, float]:
        """Compute the metrics needed for evaluation.

        Args:
            predictions_and_labels (list of pairs of NumPy arrays):
                The predictions and labels for each metric.

        Returns:
            dict:
                A dictionary with the names of the metrics as keys and the metric
                values as values.
        """
        # Iterate over the predictions, labels and associated metrics
        results = dict()
        for metric_cfg, (predictions, labels) in zip(
            self.task_config.metrics, predictions_and_labels
        ):

            # Load the metric
            metric = self._metrics[metric_cfg.name]

            # Compute the metrics. Sometimes a `RuntimeWarning` is displayed, e.g.,
            # when the predictions are all the same. We ignore this warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                score_dict = metric.compute(
                    predictions=predictions,
                    references=labels,
                    **metric_cfg.compute_kwargs,
                )

            # Add scores to the `results` dictionary
            if score_dict is not None:
                results[metric_cfg.name] = score_dict[metric_cfg.results_key]

        # Return the results
        return results

    def _prepare_predictions_and_labels(
        self,
        predictions: np.ndarray,
        dataset: Dataset,
        prepared_dataset: Dataset,
        **kwargs,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Prepare predictions and labels for output.

        Args:
            predictions (NumPy array):
                The predictions of the model.
            dataset (Dataset):
                The raw dataset.
            prepared_dataset (Dataset):
                The prepared dataset.
            kwargs:
                Extra keyword arguments containing objects used in preparing the
                predictions and labels.

        Returns:
            list of pairs of NumPy arrays:
                The prepared predictions and labels. Each list entry is a pair of NumPy
                arrays associated with each metric, with the first array being the
                predictions and the second array being the labels. If the list only
                contains one element and multiple metrics are present, then the same
                predictions and labels will be used for all the metrics.
        """
        # Collapse the logits into single predictions for every sample
        if any(
            predictions.dtype == dtype for dtype in {np.float16, np.float32, np.float64}
        ):
            predictions = np.argmax(predictions, axis=-1)

        # Extract labels from dataset
        labels = np.asarray(prepared_dataset["labels"])

        # Return the predictions and labels
        return [(predictions, labels)]

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def _load_data(self) -> Dataset:
        """Load the dataset.

        Returns:
            Dataset:
                The dataset.

        Raises:
            InvalidEvaluation:
                If the split names specified are incorrect.
        """
        return load_dataset(  # type: ignore
            path=self.task_config.huggingface_id,
            name=self.task_config.huggingface_subset,
            use_auth_token=self.evaluation_config.use_auth_token,
            cache_dir=self.evaluation_config.cache_dir,
            split=self.task_config.test_name,
        )

    @abstractmethod
    def _preprocess_data(self, dataset: Dataset, framework: str, **kwargs) -> Dataset:
        """Preprocess the data.

        Args:
            dataset (Dataset):
                The dataset.
            framework (str):
                The framework of the model.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.

        Returns:
            Hugging Face Dataset:
                The preprocessed dataset.
        """
        pass

    @abstractmethod
    def _load_data_collator(self, tokenizer: PreTrainedTokenizerBase):
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (Hugging Face tokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not used in the
                initialisation of the data collator. Defaults to None.

        Returns:
            Hugging Face data collator:
                The data collator.
        """
        pass
