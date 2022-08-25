"""Abstract Task class."""

import logging
import random
import subprocess
import warnings
from abc import ABC, abstractmethod
from subprocess import CalledProcessError
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import spacy
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .co2 import get_carbon_tracker
from .config import EvaluationConfig, ModelConfig, TaskConfig
from .exceptions import (
    InvalidArchitectureForTask,
    InvalidEvaluation,
    InvalidFramework,
    ModelFetchFailed,
    PreprocessingFailed,
    UnsupportedModelType,
    WrongFeatureColumnName,
)
from .hf_hub import get_model_config
from .metric_configs import EMISSIONS, POWER
from .scoring import log_scores
from .utils import clear_memory, enforce_reproducibility, is_module_installed

# Set up a logger
logger = logging.getLogger(__name__)


# Ignore warnings from spaCy. This has to be called after the import,
# as the __init__.py file of spaCy sets the warning levels of spaCy
# warning W036
warnings.filterwarnings("ignore", module="spacy*")


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

    def evaluate(self, model_id: str) -> Dict[str, Dict[str, float]]:
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
        model_dict = self._load_model(model_config=model_config)

        # Prepare carbon tracker
        if self.evaluation_config.track_carbon_emissions:
            self.carbon_tracker = get_carbon_tracker(
                task_name=self.task_config.name,
                country_iso_code=self.evaluation_config.country_iso_code,
                verbose=self.evaluation_config.verbose,
            )

        # Load the dataset dictinoary
        dataset_dict = self._load_data()

        # Process the datasets
        dataset_dict = self._process_data(dataset_dict)

        # Extract the dataset splits
        test = dataset_dict["test"]

        # Remove empty examples from the datasets
        try:
            test = test.filter(
                lambda x: len(x[self.task_config.feature_column_name]) > 0
            )
        except KeyError:
            raise WrongFeatureColumnName(self.task_config.feature_column_name)

        # Set variable with number of iterations
        num_iter = 10 if not self.evaluation_config.testing else 2

        if model_config.framework in {"pytorch", "jax"}:
            return self._evaluate_pytorch_jax(
                model_dict=model_dict,
                test=test,
                rng=rng,
                model_config=model_config,
                num_iter=num_iter,
            )

        elif model_config.framework == "spacy":
            return self._evaluate_spacy(
                model_dict=model_dict,
                test=test,
                rng=rng,
                model_config=model_config,
                num_iter=num_iter,
            )

        else:
            raise InvalidFramework(model_config.framework)

    def _evaluate_pytorch_jax(
        self,
        model_dict: dict,
        test: Dataset,
        rng: np.random.Generator,
        model_config: ModelConfig,
        num_iter: int,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a PyTorch or JAX model.

        Args:
            model_dict (dict):
                The model dictionary, with keys "model" and "tokenizer".
            test (Dataset):
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
        # Extract the model and tokenizer
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]

        # Log the number of parameters in the model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of model parameters: {num_params:,}")

        # Preprocess the datasets
        try:
            params = dict(
                framework="pytorch",
                config=model.config,
                tokenizer=tokenizer,
            )
            # Do framework specific preprocessing
            if isinstance(model, PreTrainedModel):
                test = self._preprocess_data_transformer(test, **params)
            elif isinstance(model, nn.Module):
                test = self._preprocess_data_pytorch(test, **params)  # type: ignore

        except ValueError:
            raise PreprocessingFailed()

        # If we are testing then truncate the test set
        if self.evaluation_config.testing:
            test = Dataset.from_dict(test[:4])

        # Get bootstrapped datasets
        tests = [
            Dataset.from_dict(test[rng.integers(0, len(test), len(test))])
            for _ in range(num_iter)
        ]

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
                    tests=tests,
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
        )
        return all_scores

    def _evaluate_pytorch_jax_single_iteration(
        self,
        idx: int,
        model_config: ModelConfig,
        tests: Sequence[Dataset],
        data_collator: DataCollator,
    ) -> Union[dict, Exception]:
        """Run a single iteration of a PyTorch/JAX benchmark.

        Args:
            idx (int):
                The index of the current iteration.
            model_dict (dict):
                The model dictionary, with keys "model" and "tokenizer".
            tests (list):
                A list of bootstraped test datasets.
            data_collator (DataCollator):
                The data collator.

        Returns:
            dict or Exception:
                The keys in the dict correspond to the metrics and values
                the corresponding values.
        """
        scores = list()
        return_scores = dict()
        try:
            # Set random seeds to enforce reproducibility of the randomly
            # initialised weights
            random.seed(703 + idx)
            np.random.seed(703 + idx)
            torch.manual_seed(703 + idx)
            torch.cuda.manual_seed_all(703 + idx)

            # Reinitialise a new model
            model_dict = self._load_model(model_config=model_config)
            model = model_dict["model"]

            # Get iteration data
            test = tests[idx]

            # Define batch size, which depends on whether we are testing or not
            batch_size = 2 if self.evaluation_config.testing else 32
            # Create dataloader
            dataloader = DataLoader(
                test, batch_size=batch_size, shuffle=True, collate_fn=data_collator  # type: ignore
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
            for batch in itr:

                # If we are dealing with a Hugging Face model then the `batch` is a
                # dictionary of tensors
                if isinstance(model, PreTrainedModel):
                    batch = {
                        key: value.to(self.evaluation_config.device)
                        for key, value in batch.items()
                    }
                    model_predictions = model(**batch).logits

                # Otherwise, if we are dealing with a PyTorch model then the `batch` is
                # a tensor of inputs
                elif isinstance(model, nn.Module):
                    batch = batch.to(self.evaluation_config.device)
                    model_predictions = model(batch)

                # Otherwise, we throw an error
                else:
                    raise UnsupportedModelType(str(type(model)))

                # Compute the metrics
                metrics = self._compute_metrics(
                    predictions=model_predictions,
                    labels=batch["labels"],
                    id2label=model.config.id2label,
                )

                # Append the metrics to the list of all scores
                scores.append(metrics)

            # Stop carbon emissions tracking
            if self.evaluation_config.track_carbon_emissions:
                self.carbon_tracker.stop()
                emissions_data = self.carbon_tracker.final_emissions_data
                return_scores["carbon_emissions"] = 1000 * emissions_data.emissions
                return_scores["energy_consumed"] = 1000 * emissions_data.energy_consumed

            if len(scores) > 0:
                for metric_cfg in self.task_config.metrics:
                    return_scores[metric_cfg.name] = np.mean(
                        [score[metric_cfg.name] for score in scores]
                    )
            return return_scores

        except (RuntimeError, ValueError, IndexError) as e:
            try:
                del model
            except UnboundLocalError:
                pass
            try:
                del model_dict
            except UnboundLocalError:
                pass
            clear_memory()
            return e

    def _evaluate_spacy(
        self,
        model_dict: dict,
        test: Dataset,
        rng: np.random.Generator,
        model_config: ModelConfig,
        num_iter: int,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a PyTorch or JAX model.

        Args:
            model_dict (dict):
                The model dictionary, with keys "model" and "tokenizer".
            test (Dataset):
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
        # TODO: Needs implementation
        return {"foo": {"bar": 1.0}}

    def _compute_metrics(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        id2label: Optional[list] = None,
    ) -> Dict[str, float]:
        """Compute the metrics needed for evaluation.

        Args:
            predictions (PyTorch tensor or NumPy array):
                The predictions of the model.
            labels (PyTorch tensor or NumPy array):
                The ground truth labels.
            id2label (list or None, optional):
                Conversion of indices to labels. Defaults to None.

        Returns:
            dict:
                A dictionary with the names of the metrics as keys and the metric
                values as values.
        """
        # Ensure that the predictions and labels are NumPy arrays
        if isinstance(predictions, torch.Tensor):
            predictions_np = predictions.detach().cpu().numpy()
        else:
            predictions_np = np.asarray(predictions)
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)

        # Compute the predicted classes
        if any(
            predictions_np.dtype == dtype
            for dtype in {np.float16, np.float32, np.float64}
        ):
            predictions_np = np.argmax(predictions_np, axis=-1)

        # Prepare the predictions and labels for the given task
        all_predictions_labels = self._prepare_predictions_and_labels(
            predictions=predictions_np, labels=labels_np, id2label=id2label
        )

        # If there are multiple metrics but only one pair in the
        # `all_predictions_labels` list, we copy our that entry to ensure there is a
        # pair for each metric
        if len(all_predictions_labels) == 1 and len(self.task_config.metrics) > 1:
            all_predictions_labels *= len(self.task_config.metrics)

        # Compute all the metrics
        results = dict()
        for metric_cfg, predictions_labels in zip(
            self.task_config.metrics, all_predictions_labels
        ):
            predictions, labels = predictions_labels
            metric = self._metrics[metric_cfg.name]
            score_dict = metric.compute(
                predictions=predictions,
                references=labels,
                **metric_cfg.compute_kwargs,
            )
            if score_dict is not None:
                scores = score_dict[metric_cfg.results_key]
                results[metric_cfg.name] = scores

        # Return the results
        return results

    def _prepare_predictions_and_labels(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        id2label: Optional[list] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Prepare predictions and labels for output.

        Args:
            predictions (NumPy array):
                The predictions of the model.
            labels (NumPy array):
                The ground truth labels.
            id2label (list or None, optional):
                Conversion of indices to labels. Defaults to None.

        Returns:
            list of pairs of NumPy arrays:
                The prepared predictions and labels. Each list entry is a pair of NumPy
                arrays associated with each metric, with the first array being the
                predictions and the second array being the labels. If the list only
                contains one element and multiple metrics are present, then the same
                predictions and labels will be used for all the metrics.
        """
        return [(predictions, labels)]

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def _load_data(self) -> DatasetDict:
        """Load the datasets.

        Returns:
            DatasetDict:
                A dictionary containing the 'train', 'val' and 'test' splits of the
                dataset.

        Raises:
            InvalidEvaluation:
                If the split names specified are incorrect.
        """
        # Download dataset from the Hugging Face Hub
        dataset_dict: DatasetDict
        dataset_dict = load_dataset(  # type: ignore
            path=self.task_config.huggingface_id,
            use_auth_token=self.evaluation_config.use_auth_token,
            cache_dir=self.evaluation_config.cache_dir,
        )

        # Remove all other keys than the split names
        train_name = self.task_config.train_name
        val_name = self.task_config.val_name
        test_name = self.task_config.test_name
        split_names = {
            split_name for split_name in [train_name, val_name, test_name] if split_name
        }
        try:
            dataset_dict = DatasetDict(
                {split_name: dataset_dict[split_name] for split_name in split_names}
            )
        except KeyError:
            raise InvalidEvaluation(
                f'The split names "{train_name}", "{val_name}", and '
                f'"{test_name}" for the train, validation and test split are '
                "incorrect."
            )

        # Return the dataset dictionary
        return dataset_dict

    def _process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Process the data.

        Args:
            dataset_dict (DatasetDict):
                The dataset dictionary.

        Returns:
            DatasetDict:
                The processed dataset dictionary.
        """
        return dataset_dict

    def _load_model(self, model_config: ModelConfig) -> Dict[str, Any]:
        """Load the model.

        Args:
            model_config (ModelConfig):
                The model configuration.

        Returns:
            dict:
                A dictionary containing at least the key 'model', with the value being
                the model. Can contain other objects related to the model, such as its
                tokenizer.

        Raises:
            RuntimeError:
                If the framework is not recognized.
        """
        # Ensure that the framework is installed
        from_flax = model_config.framework == "jax"

        # If the framework is JAX then change it to PyTorch, since we will convert
        # JAX models to PyTorch upon download
        if model_config.framework == "jax":
            model_config.framework = "pytorch"

        if model_config.framework == "pytorch":
            return self._load_pytorch_model(model_config, from_flax=from_flax)

        elif model_config.framework == "spacy":
            return self._load_spacy_model(model_config)

        else:
            raise InvalidFramework(model_config.framework)

    def _load_pytorch_model(
        self,
        model_config: ModelConfig,
        from_flax: bool,
    ) -> Dict[str, Any]:
        """Load a PyTorch model.

        Args:
            model_config (ModelConfig):
                The configuration of the model.
            from_flax (bool):
                Whether the model is a Flax model.

        Returns:
            dict:
                A dictionary containing at least the key 'model', with the value being
                the model. Can contain other objects related to the model, such as its
                tokenizer.
        """
        try:
            # Load the configuration of the pretrained model
            config = AutoConfig.from_pretrained(
                model_config.model_id,
                revision=model_config.revision,
                use_auth_token=self.evaluation_config.use_auth_token,
            )

            # Check whether the supertask is a valid one
            supertask = self.task_config.supertask
            self._check_supertask(
                architectures=config.architectures, supertask=supertask
            )

            # Get the model class associated with the supertask
            if supertask == "token-classification":
                model_cls = AutoModelForTokenClassification  # type: ignore
            elif supertask == "sequence-classification":
                model_cls = AutoModelForSequenceClassification  # type: ignore
            else:
                raise ValueError(f"The supertask `{supertask}` was not recognised.")

            # Load the model with the correct model class
            model = model_cls.from_pretrained(
                model_config.model_id,
                revision=model_config.revision,
                use_auth_token=self.evaluation_config.use_auth_token,
                config=config,
                cache_dir=self.evaluation_config.cache_dir,
                from_flax=from_flax,
            )

        # If an error occured then throw an informative exception
        except (OSError, ValueError):
            raise InvalidEvaluation(
                f"The model {model_config.model_id} either does not have a frameworks "
                "registered, or it is a private model. If it is a private model then "
                "enable the `--use-auth-token` flag and make  sure that you are "
                "logged in to the Hub via the `huggingface-cli login` command."
            )

        # Ensure that the labels of the model are consistent with the labels of the
        # dataset
        self._adjust_label_ids(model=model, model_config=model_config)

        # If the model is a subclass of a RoBERTa model then we have to add a prefix
        # space to the tokens, by the way the model is constructed.
        m_id = model_config.model_id
        prefix = "Roberta" in type(model).__name__
        params = dict(use_fast=True, add_prefix_space=prefix)
        tokenizer = AutoTokenizer.from_pretrained(
            m_id,
            revision=model_config.revision,
            use_auth_token=self.evaluation_config.use_auth_token,
            **params,
        )

        # Set the maximal length of the tokenizer to the model's maximal length.
        # This is required for proper truncation
        if (
            not hasattr(tokenizer, "model_max_length")
            or tokenizer.model_max_length > 1_000
        ):

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
        model.to(self.evaluation_config.device)

        return dict(model=model, tokenizer=tokenizer)

    def _load_spacy_model(self, model_config: ModelConfig) -> Dict[str, Any]:
        """Load a spaCy model.

        Args:
            model_config (ModelConfig):
                The configuration of the model.

        Returns:
            dict:
                A dictionary containing at least the key 'model', with the value being
                the model. Can contain other objects related to the model, such as its
                tokenizer.
        """
        local_model_id = model_config.model_id.split("/")[-1]

        # Download the model if it has not already been so
        try:
            if not is_module_installed(local_model_id):
                url = (
                    f"https://huggingface.co/{model_config.model_id}/resolve/main/"
                    f"{local_model_id}-any-py3-none-any.whl"
                )
                subprocess.run(["pip3", "install", url])

        except CalledProcessError as e:
            raise ModelFetchFailed(model_id=local_model_id, error_msg=e.output)

        # Load the model
        try:
            model = spacy.load(local_model_id)
        except OSError as e:
            raise ModelFetchFailed(
                model_id=model_config.model_id,
                error_msg=str(e),
                message=(
                    f"Download of {model_config.model_id} failed, with "
                    f"the following error message: {str(e)}."
                ),
            )
        return dict(model=model)

    def _check_supertask(self, architectures: Sequence[str], supertask: str):
        """Checks if the supertask corresponds to the architectures, by looking for the
        search_str.

        Args:
            architectures (list of str):
                The model architecture names.
            supertask (str):
                The supertask associated to a task, e.g. text-classification.

        Raises:
            InvalidArchitectureForTask:
                If the search_str is not found in any of the architectures.
        """
        # Convert the supertask into a search string, by converting kebab case to title
        # case; e.g., text-classification -> TextClassification
        search_str = "".join(word.title() for word in supertask.split("-"))

        # Create boolean variable that checks if the supertask exists among the
        # available architectures
        supertask_is_an_architecture = any(search_str in arc for arc in architectures)

        # If the supertask is not an architecture, raise an error
        if not supertask_is_an_architecture:
            raise InvalidArchitectureForTask(
                architectures=architectures, supertask=supertask
            )

    def _adjust_label_ids(
        self,
        model: nn.Module,
        model_config: ModelConfig,
    ) -> nn.Module:
        """Adjust the label ids of the model to match the dataset.

        Args:
            model (PyTorch Module):
                The model to adjust the label ids of.
            model_config (ModelConfig):
                The model configuration.

        Returns:
            PyTorch Model:
                The model with adjusted label ids.
        """
        # TODO: Only a placeholder for now
        return model

    @abstractmethod
    def _preprocess_data_pytorch(self, dataset: Dataset, **kwargs) -> list:
        """Preprocess a dataset by tokenizing and aligning the labels.

        For use by a PyTorch model.

        Args:
            dataset (Hugging Face dataset):
                The dataset to preprocess.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.

        Returns:
            list of lists:
                Every list element represents the tokenised data for the corresponding
                example.
        """
        pass

    @abstractmethod
    def _preprocess_data_transformer(
        self, dataset: Dataset, framework: str, **kwargs
    ) -> Dataset:
        """Process the data for use by a transformer model.

        For use by a transformer model.

        Args:
            dataset_dict (DatasetDict):
                The dataset dictionary.

        Returns:
            DatasetDict:
                The processed dataset dictionary.
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
