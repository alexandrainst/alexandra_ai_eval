"""Abstract Task class."""

import logging
import random
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import evaluate as evaluate_hf
import numpy as np
import torch
import torch.nn as nn
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from spacy.language import Language
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .co2 import get_carbon_tracker
from .config import EvaluationConfig, ModelConfig, TaskConfig
from .enums import Framework
from .exceptions import (
    InvalidEvaluation,
    InvalidFramework,
    ModelNotTrainedForTask,
    MPSFallbackNotEnabled,
    PreprocessingFailed,
    UnsupportedModelType,
    WrongFeatureColumnName,
)
from .metric_configs import EMISSIONS, POWER
from .model_loading import get_model_config, load_model
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
            metric_cfg.name: evaluate_hf.load(metric_cfg.huggingface_id)
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

        Raises:
            WrongFeatureColumnName:
                If the feature column name specified in the task configuration is
                incorrect.
            InvalidEvaluation:
                If an error occurs during evaluation.
        """
        # Fetch the model config
        model_config = get_model_config(
            model_id=model_id,
            task_config=self.task_config,
            evaluation_config=self.evaluation_config,
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
                country_code=self.evaluation_config.country_code,
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

        # Extract the model and tokenizer
        model = model_dict["model"]
        tokenizer = model_dict.get("tokenizer")
        processor = model_dict.get("processor")

        # Log the number of parameters in the model
        if model_config.framework == Framework.PYTORCH:
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
        prepared_datasets = [
            self._preprocess_data(
                bootstrapped_dataset,
                framework=model_config.framework,
                model_config=model_config,
                tokenizer=tokenizer,
            )
            for bootstrapped_dataset in bootstrapped_datasets
        ]

        # Set up progress bar
        if self.evaluation_config.progress_bar:
            itr = tqdm(range(num_iter), desc="Evaluating")
        else:
            itr = range(num_iter)

        scores = list()
        for idx in itr:
            while True:
                test_itr_scores_or_err = self._evaluate_single_iteration(
                    idx=idx,
                    model=model,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    processor=processor,
                    framework=model_config.framework,
                    dataset=bootstrapped_datasets[idx],
                    prepared_dataset=prepared_datasets[idx],
                )

                # If the iteration was successful then break the while-loop
                if isinstance(test_itr_scores_or_err, dict):
                    break

                # Otherwise we encountered an error
                else:
                    raise InvalidEvaluation(
                        f"An unknown error occurred during the evaluation of the {idx} "
                        "iteration. The error message returned was: "
                        f"{str(test_itr_scores_or_err)}"
                    )

            scores.append(test_itr_scores_or_err)

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

    def _evaluate_single_iteration(
        self,
        idx: int,
        model: Union[nn.Module, Language],
        model_config: ModelConfig,
        tokenizer: Optional[PreTrainedTokenizerBase],
        processor: Optional[AutoProcessor],
        dataset: Dataset,
        prepared_dataset: Dataset,
        framework: Framework,
    ) -> Union[dict, Exception]:
        """Run a single iteration of a PyTorch/JAX benchmark.

        Args:
            idx (int):
                The index of the current iteration.
            model (PyTorch module or spaCy Language):
                The model.
            model_config (ModelConfig):
                The model configuration.
            tokenizer (Hugging Face tokenizer or None):
                The tokenizer, or None if the model does not require a tokenizer.
            dataset (Dataset):
                The raw test dataset.
            prepared_dataset (Dataset):
                The preprocessed test dataset.
            framework (Framework):
                The model framework.

        Returns:
            dict or Exception:
                The keys in the dict correspond to the metrics and values
                the corresponding values.

        Raises:
            ModelNotTrainedForTask:
                If the model is not trained for the task.
            MPSFallbackNotEnabled:
                If the MPS device is used, but the CPU fallback is not enabled.
        """
        try:
            # Set random seeds to enforce reproducibility of the randomly
            # initialised weights
            random.seed(703 + idx)
            np.random.seed(703 + idx)
            torch.manual_seed(703 + idx)
            torch.cuda.manual_seed_all(703 + idx)

            # Define batch size, which depends on whether we are testing or not
            batch_size = 2 if self.evaluation_config.testing else 32

            # Start carbon emissions tracking
            if self.evaluation_config.track_carbon_emissions:
                self.carbon_tracker.start()

            # Get model predictions
            model_predictions = self._get_model_predictions(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                prepared_dataset=prepared_dataset,
                batch_size=batch_size,
                framework=framework,
            )

            # Extract attributes if they are available
            try:
                model_id2label = model_config.id2label  # type: ignore[attr-defined]
            except AttributeError:
                model_id2label = None
            try:
                cls_token_index = tokenizer.cls_token_id  # type: ignore[union-attr]
            except AttributeError:
                cls_token_index = None

            # Perform post-processing of predictions
            prepared_predictions_and_labels = self._prepare_predictions_and_labels(
                predictions=model_predictions,
                dataset=dataset,
                prepared_dataset=prepared_dataset,
                model_id2label=model_id2label,
                cls_token_index=cls_token_index,
                processor=processor,
            )

            # If there are multiple metrics but only one pair in the
            # `all_predictions_labels` list, we copy our that entry to ensure there is a
            # pair for each metric
            if (
                len(prepared_predictions_and_labels) == 1
                and len(self.task_config.metrics) > 1
            ):
                prepared_predictions_and_labels *= len(self.task_config.metrics)

            # In the first iteration we do a check to see if the model outputs fit the
            # expected format. If not, we raise an exception.
            if idx == 0:
                if not self._check_if_model_is_trained_for_task(
                    model_predictions=model_predictions
                ):
                    raise ModelNotTrainedForTask(task=self.task_config.name)

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
            clear_memory()

            # Return the error if it wasn't caught by the above conditionals
            return e

    def _compute_metrics(
        self,
        predictions_and_labels: List[Tuple[list, list]],
    ) -> Dict[str, float]:
        """Compute the metrics needed for evaluation.

        Args:
            predictions_and_labels (list of pairs of lists):
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
                # Some metrics return a single value, others a dictionary. We
                # standardise this by always returning a dictionary.
                if isinstance(score_dict, float):
                    score_dict = {metric_cfg.huggingface_id: score_dict}

            # Add scores to the `results` dictionary
            if score_dict is not None:
                results[metric_cfg.name] = score_dict[metric_cfg.results_key]

        # Return the results
        return results

    def _load_data(self) -> Dataset:
        """Load the dataset.

        Returns:
            Dataset:
                The dataset.

        Raises:
            InvalidEvaluation:
                If the split names specified are incorrect.
        """
        return load_dataset(
            path=self.task_config.huggingface_id,
            name=self.task_config.huggingface_subset,
            use_auth_token=self.evaluation_config.use_auth_token,
            cache_dir=self.evaluation_config.cache_dir,
            split=self.task_config.test_name,
        )

    def _prepare_pytorch_batch(self, batch: dict, input_modality: str) -> dict:
        """Prepare a batch for the PyTorch model.

        Args:
            batch (dict):
                The batch.
            input_modality (str):
                The input modality, can be 'audio' or 'text'.

        Returns:
            dict:
                The prepared batch.
        """
        # Move the tensors to the correct device
        batch = {
            key: value.to(self.evaluation_config.device) for key, value in batch.items()
        }

        # Create a view of the batch with only desired features
        if input_modality == "text":
            accepted_transformer_features = [
                "input_ids",
                "attention_mask",
                "token_type_ids",
            ]
        # Whisper takes "input_features", while Wav2Vec2 takes "input_values"
        elif input_modality == "audio":
            accepted_transformer_features = ["input_features", "input_values"]

        batch = {
            key: value
            for key, value in batch.items()
            if key in accepted_transformer_features
        }

        # Return the prepared batch
        return batch

    def _get_model_predictions(
        self,
        model: Union[nn.Module, Language],
        tokenizer: Optional[PreTrainedTokenizerBase],
        processor: Optional[AutoProcessor],
        prepared_dataset: Dataset,
        batch_size: int,
        framework: Framework,
    ) -> list:
        """Get the predictions of the model.

        Args:
            model (torch.nn.Module or Language):
                The model.
            tokenizer (PreTrainedTokenizerBase or None):
                The tokenizer. Can be None if the model does not use a tokenizer.
            prepared_dataset (Dataset):
                The prepared dataset.
            batch_size (int):
                The batch size.
            framework (Framework):
                The framework.

        Returns:
            list:
                The model predictions.

        Raises:
            InvalidFramework:
                If the framework is not a supported framework.
            ValueError:
                If the model predictions are not in the right format.
            UnsupportedModelType:
                If the model type is not supported.
        """
        if framework == Framework.PYTORCH:

            # Load the data collator
            if not isinstance(processor, PreTrainedTokenizerBase):
                data_collator = self._load_data_collator(
                    tokenizer_or_processor=processor
                )
            else:
                data_collator = self._load_data_collator(
                    tokenizer_or_processor=tokenizer
                )

            dataloader = DataLoader(
                prepared_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )

            # Create progress bar
            if self.evaluation_config.progress_bar:
                itr = tqdm(iterable=dataloader, desc="Evaluating", leave=False)
            else:
                itr = dataloader

            all_predictions = list()
            for batch in itr:

                # Define input modality, used for preparing the batch
                # TODO: should probably be picked up from model somehow, or
                # be part of the task config.
                if self.task_config.name == "automatic-speech-recognition":
                    input_modality = "audio"
                else:
                    input_modality = "text"

                # Prepare the batch
                batch = self._prepare_pytorch_batch(
                    batch, input_modality=input_modality
                )

                # If we are dealing with a Hugging Face model then we will use the
                # entire batch dictionary
                if isinstance(model, PreTrainedModel):

                    # Get the model predictions
                    with torch.no_grad():
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                action="ignore", category=UserWarning
                            )
                            # Whisper models have a different API compared to even other
                            # ASR models so we handle it in a specific case here.
                            if isinstance(model, WhisperForConditionalGeneration):
                                forced_decoder_ids = None
                                if processor is not None:
                                    forced_decoder_ids = (
                                        processor.get_decoder_prompt_ids(
                                            language="da", task="transcribe"
                                        )
                                    )
                                model_predictions = model.generate(
                                    input_features=batch["input_features"],
                                    forced_decoder_ids=forced_decoder_ids,
                                )
                            else:
                                model_predictions = model(**batch)

                    # If we are dealing with a classification model then we will take
                    # the logits
                    if hasattr(model_predictions, "logits"):
                        model_predictions = model_predictions.logits

                    # If we are dealing with a question answering model then we will
                    # take the start and end logits and merge them
                    elif all(
                        [
                            hasattr(model_predictions, "start_logits"),
                            hasattr(model_predictions, "end_logits"),
                        ]
                    ):
                        model_predictions = torch.stack(
                            [
                                model_predictions.start_logits,
                                model_predictions.end_logits,
                            ],
                            dim=-1,
                        )
                    # In case of ASR, we recieve a tensor, and we if we do not, then
                    # we throw an error.
                    elif not isinstance(model_predictions, torch.Tensor):
                        raise ValueError(
                            "The model predictions are not in the correct format. "
                            f"Received outputs with keys {model_predictions.keys()}"
                        )

                # If we are dealing with a PyTorch model, then we will only use the
                # input_ids
                elif isinstance(model, nn.Module):
                    with torch.no_grad():
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                action="ignore", category=UserWarning
                            )
                            model_predictions = model(batch["input_ids"])

                # Otherwise, we throw an error
                else:
                    model_type = str(type(model))
                    raise UnsupportedModelType(model_type=model_type)

                # Move the predictions back to the CPU and convert it to a NumPy array
                model_predictions = model_predictions.cpu().numpy()

                # Collect predictions
                all_predictions.extend(model_predictions)

            return all_predictions

        elif framework == Framework.SPACY:

            # Create progress bar
            if self.evaluation_config.progress_bar:
                itr = tqdm(
                    prepared_dataset[self.task_config.feature_column_names[0]],
                    desc="Evaluating model",
                    leave=False,
                )
            else:
                itr = prepared_dataset[self.task_config.feature_column_names[0]]

            # Apply the model to the dataset
            processed = model.pipe(itr, batch_size=batch_size)

            # Extract the predictions using a task-specific function
            predictions = map(
                self._extract_spacy_predictions,
                zip(prepared_dataset["tokens"], processed),
            )

            return list(predictions)

        else:
            raise InvalidFramework(framework=framework)

    def _preprocess_data(
        self, dataset: Dataset, framework: Framework, **kwargs
    ) -> Dataset:
        """Preprocess the data.

        Args:
            dataset (Dataset):
                The dataset.
            framework (Framework):
                The framework of the model.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.

        Returns:
            Hugging Face Dataset:
                The preprocessed dataset.

        Raises:
            InvalidFramework:
                If the framework is not a supported framework.
            PreprocessingFailed:
                If the preprocessing failed.
        """
        try:
            if framework == Framework.PYTORCH:

                preprocess_fn = partial(
                    self._pytorch_preprocess_fn,
                    tokenizer=kwargs["tokenizer"],
                    model_config=kwargs["model_config"],
                    task_config=self.task_config,
                )
                preprocessed = dataset.map(
                    preprocess_fn,
                    batched=True,
                    remove_columns=dataset.column_names,
                )
                return preprocessed

            elif framework == Framework.SPACY:
                return dataset.map(self._spacy_preprocess_fn, batched=True)

            else:
                raise InvalidFramework(framework=framework)

        except ValueError:
            raise PreprocessingFailed()

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    @abstractmethod
    def _prepare_predictions_and_labels(
        self,
        predictions: Sequence,
        dataset: Dataset,
        prepared_dataset: Dataset,
        **kwargs,
    ) -> List[Tuple[list, list]]:
        """Prepare predictions and labels for output.

        Args:
            predictions (sequence of either ints or floats):
                The predictions of the model.
            dataset (Dataset):
                The raw dataset.
            prepared_dataset (Dataset):
                The prepared dataset.
            kwargs:
                Extra keyword arguments containing objects used in preparing the
                predictions and labels.

        Returns:
            list of pairs of lists:
                The prepared predictions and labels.
        """
        pass

    @abstractmethod
    def _spacy_preprocess_fn(self, examples: BatchEncoding) -> BatchEncoding:
        """Preprocess the data for spaCy.

        Args:
            examples (BatchEncoding):
                The examples to preprocess.

        Returns:
            BatchEncoding:
                The preprocessed examples.
        """
        pass

    @abstractmethod
    def _pytorch_preprocess_fn(
        self,
        examples: BatchEncoding,
        tokenizer: PreTrainedTokenizerBase,
        model_config: ModelConfig,
        task_config: TaskConfig,
    ) -> BatchEncoding:
        """Preprocess the data for PyTorch.

        Args:
            examples (BatchEncoding):
                The examples to preprocess.
            tokenizer (Hugging Face tokenizer):
                The tokenizer.
            model_config (ModelConfig):
                The model configuration.
            task_config (TaskConfig):
                The task configuration.

        Returns:
            dict:
                The preprocessed examples.
        """
        pass

    @abstractmethod
    def _extract_spacy_predictions(self, tokens_processed: tuple) -> list:
        """Helper function that extracts the predictions from a SpaCy model.

        Aside from extracting the predictions from the model, it also aligns the
        predictions with the gold tokens, in case the SpaCy tokenizer tokenizes the
        text different from those.

        Args:
            tokens_processed (tuple):
                A pair of the labels, being a list of strings, and the SpaCy processed
                document, being a Spacy `Doc` instance.

        Returns:
            list:
                A list of predictions for each token, of the same length as the gold
                tokens (first entry of `tokens_processed`).
        """
        pass

    @abstractmethod
    def _load_data_collator(
        self, tokenizer_or_processor: Union[PreTrainedTokenizerBase, AutoProcessor]
    ) -> DataCollator:
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer_or_processor (Hugging Face tokenizer or AutoProcessor):
                A pretrained tokenizer or processor.

        Returns:
            Hugging Face DataCollator:
                The data collator.
        """
        pass

    @abstractmethod
    def _check_if_model_is_trained_for_task(self, model_predictions: list) -> bool:
        """Check if the model is trained for the task.

        Args:
            model_predictions (list):
                The model predictions.

        Returns:
            bool:
                Whether the model is trained for the task.
        """
        pass
