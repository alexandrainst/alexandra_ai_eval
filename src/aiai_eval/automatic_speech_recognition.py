"""Class for automatic speech recognition tasks."""

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union
from unicodedata import normalize

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from numpy.typing import NDArray
from transformers import Wav2Vec2Processor
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .config import TaskConfig
from .exceptions import FrameworkCannotHandleTask
from .task import Task


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (Wav2Vec2Processor)
            The processor used for proccessing the data.
        padding (bool, str or PaddingStrategy, optional):
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
    padding: Union[bool, str] = True

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        """Collate the features.
        Args:
            features (list of dict):
                A list of feature dicts.
        Returns:
            dict:
                A dictionary of the collated features.
        """
        # Get sampling rate
        sampling_rate = self.processor.feature_extractor.sampling_rate

        # Split inputs and labels since they have to be of different lenghts
        # and need different padding methods
        input_features = [
            {
                "input_features": self.processor(
                    feature["input_values"]["array"],
                    sampling_rate=sampling_rate,
                ).input_features[0]
            }
            for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Create batch from input_features
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Process labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        non_one_entries = labels_batch.attention_mask.ne(1)
        labels = labels_batch["input_ids"].masked_fill(non_one_entries, -100)

        # Update the batch labels
        batch["labels"] = labels

        # Return the updated batch
        return batch


class AutomaticSpeechRecognition(Task):
    """Automatic Speech Recognition task.

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

    def _pytorch_preprocess_fn(
        self,
        examples: BatchEncoding,
        tokenizer: PreTrainedTokenizerBase,
        model_config: PretrainedConfig,
        task_config: TaskConfig,
    ) -> BatchEncoding:
        def clean_transcription(doc: str) -> str:
            """Cleans the transcription of a document.
            Args:
                doc (str):
                    A document to be cleaned.
            Returns:
                str:
                    The cleaned document.
            """
            # NFKC normalize the transcriptions
            doc = normalize("NFKC", doc)

            # Remove punctuation
            regex = r"[\[\]\{\}\(\)\,\?\.\!\-\—\–\;\:\"\“\'\’\%\”\�\•\n\r\⁄\’]"
            doc = re.sub(regex, "", doc)

            # Make the transcription lowercase and strip whitespace
            doc = doc.lower().strip()

            return doc

        # Preprocess the transcriptions
        examples[task_config.label_column_name] = [
            clean_transcription(example)
            for example in examples[task_config.label_column_name]
        ]

        # Create labels column
        examples["labels"] = tokenizer(
            list(examples[task_config.label_column_name]),
            truncation=True,
        )["input_ids"]

        # If there is more than on feature column list raise an exception
        if len(task_config.feature_column_names) != 1:
            raise ValueError(
                "Only one feature column is supported, for the Automatic Speech Recognition task."
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
        self, tokenizer_or_processor: Union[PreTrainedTokenizerBase, AutoProcessor]
    ) -> DataCollatorCTCWithPadding:
        return DataCollatorCTCWithPadding(processor=tokenizer_or_processor)

    def _prepare_predictions_and_labels(
        self,
        predictions: Sequence,
        dataset: Dataset,
        prepared_dataset: Dataset,
        **kwargs,
    ) -> List[Tuple[list, list]]:
        # Get processor
        processor = kwargs["processor"]

        # Make predictions into a numpy array
        predictions_arr: NDArray[np.int_] = np.array(predictions)

        # Get the padding token
        pad_token = processor.tokenizer.pad_token_id

        # Set the ground truth labels with label id -100 to be the padding token id
        predictions_arr[predictions == -100] = pad_token

        # Decode the predictions
        predictions_str = processor.batch_decode(
            predictions_arr, skip_special_tokens=True
        )

        # Decode the labels
        label_str = processor.tokenizer.batch_decode(prepared_dataset["labels"])

        return list(zip([predictions_str], [label_str]))

    def _check_if_model_is_trained_for_task(self, model_predictions: list) -> bool:
        return True
