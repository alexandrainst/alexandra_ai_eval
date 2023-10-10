"""Class for automatic speech recognition tasks."""

from dataclasses import dataclass

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, WhisperProcessor
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .config import TaskConfig
from .exceptions import FrameworkCannotHandleTask
from .task import Task


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that will dynamically pad the inputs received.

    Args:
        processor:
            The processor used for proccessing the data.
        padding:
            Select a strategy to pad the returned sequences (according to the
            model's padding side and padding index) among:
            * True or 'longest':
                Pad to the longest sequence in the batch (or no padding if only
                a single sequence if provided).
            * 'max_length':
                Pad to a maximum length specified with the argument max_length
                or to the maximum acceptable input length for the model if that
                argument is not provided.
            * False or 'do_not_pad':
                No padding (i.e., can output a batch with sequences of
                different lengths).
            Defaults to True.
    """

    processor: Wav2Vec2Processor
    padding: bool | str = True

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate the features.

        Args:
            features:
                A list of feature dicts.

        Returns:
            A dictionary of the collated features.
        """
        sampling_rate = self.processor.feature_extractor.sampling_rate

        # Whisper and Wav2Vec2 have different input APIs which we need to take into
        # account
        if isinstance(self.processor, WhisperProcessor):
            input_features = [
                {
                    "input_features": self.processor(
                        feature["input_values"]["array"],
                        sampling_rate=sampling_rate,
                    ).input_features[0]
                }
                for feature in features
            ]
        elif isinstance(self.processor, Wav2Vec2ProcessorWithLM) or isinstance(
            self.processor, Wav2Vec2Processor
        ):
            input_features = [
                {
                    "input_values": self.processor(
                        feature["input_values"]["array"],
                        sampling_rate=sampling_rate,
                    ).input_values[0]
                }
                for feature in features
            ]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        return batch


class AutomaticSpeechRecognition(Task):
    """Automatic Speech Recognition task.

    Args:
        task_config:
            The configuration of the task.
        evaluation_config:
            The configuration of the evaluation.

    Attributes:
        task_config:
            The configuration of the task.
        evaluation_config:
            The configuration of the evaluation.
    """

    def _pytorch_preprocess_fn(
        self,
        examples: BatchEncoding,
        tokenizer: PreTrainedTokenizerBase,
        model_config: PretrainedConfig,
        task_config: TaskConfig,
    ) -> BatchEncoding:
        examples["labels"] = examples[task_config.label_column_name]

        # If there is more than on feature column list raise an exception
        if len(task_config.feature_column_names) != 1:
            raise ValueError(
                "Only one feature column is supported, for the Automatic Speech "
                "Recognition task."
            )

        # Rename the feature column to input_values
        examples["input_values"] = examples.pop(task_config.feature_column_names[0])
        return examples

    def _spacy_preprocess_fn(self, examples: dict) -> dict:
        raise FrameworkCannotHandleTask(
            framework="spaCy", task=self.task_config.pretty_name
        )

    def _extract_spacy_predictions(self, tokens_processed: tuple) -> list:
        raise FrameworkCannotHandleTask(
            framework="spaCy", task=self.task_config.pretty_name
        )

    def _load_data_collator(
        self, tokenizer_or_processor: PreTrainedTokenizerBase | AutoProcessor
    ) -> DataCollatorCTCWithPadding:
        return DataCollatorCTCWithPadding(processor=tokenizer_or_processor)

    def _prepare_predictions_and_labels(
        self,
        predictions: list,
        dataset: Dataset,
        prepared_dataset: Dataset,
        **kwargs,
    ) -> list[tuple[list, list]]:
        processor = kwargs["processor"]

        # Decode the predictions, Whisper has a different API
        if isinstance(processor, Wav2Vec2ProcessorWithLM):
            predictions_str = processor.batch_decode(np.array(predictions)).text

        elif isinstance(processor, Wav2Vec2Processor):
            predictions_ids = np.argmax(predictions, axis=-1)
            predictions_str = processor.batch_decode(predictions_ids)

        elif isinstance(processor, WhisperProcessor):
            predictions_str = processor.batch_decode(
                predictions, skip_special_tokens=True
            )

        label_str = prepared_dataset["labels"]

        return list(zip([predictions_str], [label_str]))

    def _check_if_model_is_trained_for_task(self, model_predictions: list) -> bool:
        return True
