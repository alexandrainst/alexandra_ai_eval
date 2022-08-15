"""Abstract Task class."""

import logging
import random
import subprocess
import warnings
from abc import ABC, abstractmethod
from functools import partial
from subprocess import CalledProcessError
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
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
    PreTrainedTokenizerBase,
)

from .config import DatasetTask, EvaluationConfig, ModelConfig
from .exceptions import InvalidEvaluation, InvalidFramework, ModelFetchFailed
from .hf_hub import get_model_config
from .utils import (
    clear_memory,
    enforce_reproducibility,
    is_module_installed,
    log_scores,
)

logger = logging.getLogger(__name__)


class EvaluationTask(ABC):
    """Abstract evaluation task class.

    Args:
        dataset_task (DatasetTask):
            The configuration of the dataset task.
        evaluation_config (EvaluationConfig):
            The configuration of the evaluation.

    Attributes:
        dataset_task (DatasetTask):
            The configuration of the dataset task.
        evaluation_config (EvaluationConfig):
            The configuration of the evaluation.
    """

    def __init__(self, dataset_task: DatasetTask, evaluation_config: EvaluationConfig):
        """Initialise the dataset.

        Args:
            dataset_task (DatasetTask):
                The configuration for the dataset.
            evaluation_config (EvaluationConfig):
                The configuration for the benchmark.
        """
        self.dataset_task = dataset_task
        self.evaluation_config = evaluation_config
        self._metrics = {
            metric_cfg.name: load_metric(metric_cfg.huggingface_id)
            for metric_cfg in dataset_task.metrics
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

        # Load the dataset dictinoary
        dataset_dict = self._load_data()

        # Process the datasets
        dataset_dict = self._process_data(dataset_dict)

        # Extract the dataset splits
        test = dataset_dict["test"]

        # Remove empty examples from the datasets
        try:
            test = test.filter(lambda x: len(x["tokens"]) > 0)
        except KeyError:
            try:
                test = test.filter(lambda x: len(x["doc"]) > 0)
            except KeyError:
                message = "Removal of empty examples was attempted, but failed."
                warnings.warn(message)

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
            raise RuntimeError(
                f'The framework "{model_config.framework}" is not supported!'
            )

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
                dataset_task=self.dataset_task,
            )
            test = self._preprocess_data(test, **params)
        except ValueError:
            raise InvalidEvaluation("Preprocessing of the dataset could not be done.")

        # If we are testing then truncate the test set
        if self.evaluation_config.testing:
            test = Dataset.from_dict(test[:128])

        # Get bootstrapped datasets
        tests = [
            Dataset.from_dict(test[rng.integers(0, len(test), len(test))])
            for _ in range(num_iter)
        ]

        # Set up progress bar
        if self.evaluation_config.progress_bar:
            itr = tqdm(range(num_iter), desc="Benchmarking")
        else:
            itr = range(num_iter)

        # Load the data collator
        data_collator = self._load_data_collator(tokenizer)

        scores = []
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
                    raise InvalidEvaluation(str(test_itr_scores))

            scores.append(test_itr_scores)

        # Log scores
        all_scores = log_scores(
            dataset_name=self.dataset_task.pretty_dataset_name,
            metric_configs=self.dataset_task.metrics,
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
        scores = []
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

            # Initialise compute_metrics function
            compute_metrics = partial(
                self._compute_metrics, id2label=model.config.id2label
            )

            # Get iteration data
            test = tests[idx]

            # Create dataloader
            dataloader = DataLoader(
                test, batch_size=32, shuffle=True, collate_fn=data_collator  # type: ignore
            )

            # Get model predictions
            for batch in dataloader:
                model_predictions = model(**batch).logits

                # Compute metrics
                scores.append(compute_metrics((model_predictions, batch["labels"])))
                break

            # Aggregate scores from batches
            return_scores = {}
            for metric in self.dataset_task.metrics:
                if len(scores):
                    return_scores[metric.name] = np.average(
                        [score[metric.name] for score in scores]
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
        return {"foo": {"bar": 1.0}}

    def _compute_metrics(
        self, predictions_and_labels: tuple, id2label: Optional[list] = None
    ) -> Dict[str, float]:
        """Compute the metrics needed for evaluation.
        Args:
            predictions_and_labels (pair of arrays):
                The first array contains the probability predictions and the second
                array contains the true labels.
            id2label (list or None, optional):
                Conversion of indices to labels. Defaults to None.
        Returns:
            dict:
                A dictionary with the names of the metrics as keys and the metric
                values as values.
        """
        predictions, labels = predictions_and_labels
        predictions = predictions.argmax(axis=-1)
        results = dict()
        for cfg in self.dataset_task.metrics:
            metric = self._metrics[cfg.name]
            score_dict = metric.compute(
                predictions=predictions,
                references=labels,
                **cfg.compute_kwargs,
            )
            if score_dict is not None:
                scores = score_dict[cfg.results_key]
                results[cfg.name] = scores
        return results

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def _load_data(self) -> DatasetDict:
        """Load the datasets.

        Returns:
            DatasetDict:
                A dictionary containing the 'train', 'val' and 'test' splits of the
                dataset.
        """
        # Download dataset from the HF Hub
        dataset_dict: DatasetDict
        dataset_dict = load_dataset(  # type: ignore
            path=self.dataset_task.huggingface_id,
            use_auth_token=self.evaluation_config.use_auth_token,
            cache_dir=self.evaluation_config.cache_dir,
        )

        # Remove all other keys than 'train', 'test', 'val'
        try:
            dataset_dict = DatasetDict(
                {
                    key: dataset_dict.get(self.dataset_task.split_names[key])
                    for key in ["train", "val", "test"]
                }
            )
        except KeyError:
            message = (
                f"`split_names`: {list(self.dataset_task.split_names.values())}, does not correspond "
                f"to found splits: {list(dataset_dict.keys())}"
            )
            raise ValueError(message)

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
            RuntimeError: If the framework is not recognized.
        """
        # Ensure that the framework is installed
        from_flax = model_config.framework == "jax"
        try:
            # If the framework is JAX then change it to PyTorch, since we will convert
            # JAX models to PyTorch upon download
            if model_config.framework == "jax":
                model_config.framework = "pytorch"

            elif model_config.framework == "spacy":
                import spacy

                # Ignore warnings from spaCy. This has to be called after the import,
                # as the __init__.py file of spaCy sets the warning levels of spaCy
                # warning W036
                warnings.filterwarnings("ignore", module="spacy*")

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"The model {model_config.model_id} is built using the spaCy "
                "framework which is not installed."
            )

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
            config = AutoConfig.from_pretrained(
                model_config.model_id,
                revision=model_config.revision,
                use_auth_token=self.evaluation_config.use_auth_token,
            )

            supertask = self.dataset_task.supertask
            if supertask == "token-classification":
                model_cls = AutoModelForTokenClassification  # type: ignore
            elif supertask == "text-classification":
                model_cls = AutoModelForSequenceClassification  # type: ignore
            else:
                raise ValueError(f"The supertask `{supertask}` was not recognised.")

            model = model_cls.from_pretrained(
                model_config.model_id,
                revision=model_config.revision,
                use_auth_token=self.evaluation_config.use_auth_token,
                config=config,
                cache_dir=self.evaluation_config.cache_dir,
                from_flax=from_flax,
            )

        except (OSError, ValueError):
            msg = (
                f"The model {model_config.model_id} either does not have a "
                "frameworks registered, or it is a private model. If it is a "
                "private model then enable the `--use-auth-token` flag and "
                "make  sure that you are logged in to the Hub via the "
                "`huggingface-cli login` command."
            )
            raise InvalidEvaluation(msg)

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
        import spacy

        # Ignore warnings from spaCy. This has to be called after the import, as the
        # __init__.py file of spaCy sets the warning levels of spaCy warning W036
        warnings.filterwarnings("ignore", module="spacy*")

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
        except OSError:
            raise InvalidEvaluation(
                f"The model {model_config.model_id} could not be installed from spaCy."
            )
        return dict(model=model)

    def _adjust_label_ids(
        self,
        model: nn.Module,
        model_config: ModelConfig,
    ) -> nn.Module:
        """Adjust the label ids of the model to match the dataset.

        Args:
            model (PyTorch Model):
                The model to adjust the label ids of.
            model_config (ModelConfig):
                The model configuration.

        Returns:
            PyTorch Model:
                The model with adjusted label ids.
        """
        # Placeholder for now.
        return model

    @abstractmethod
    def _preprocess_data(self, dataset: Dataset, framework: str, **kwargs) -> Dataset:
        """Preprocess a dataset by tokenizing and aligning the labels.
        Args:
            dataset (Hugging Face dataset):
                The dataset to preprocess.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.
        Returns:
            Hugging Face dataset: The preprocessed dataset.
        """
        pass

    @abstractmethod
    def _load_data_collator(self, tokenizer: Optional[PreTrainedTokenizerBase] = None):
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
