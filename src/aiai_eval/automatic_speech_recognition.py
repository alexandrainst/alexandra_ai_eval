"""Class for automatic speech recognition tasks."""

import re
from concurrent.futures import process
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
from unicodedata import normalize

import torch
from datasets.arrow_dataset import Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2Processor
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
        # Split inputs and labels since they have to be of different lenghts
        # and need different padding methods
        input_features = [
            {
                "input_values": self.processor(
                    feature["input_features"],
                ).input_values[0]
            }
            for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Process audio
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Process labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
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
        processor: Optional[AutoProcessor],
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

        def clean_audio(
            feature_extractor: AutoFeatureExtractor, examples: BatchEncoding
        ) -> BatchEncoding:
            """Cleans the audio of a document.
            Args:
                feature_extractor (AutoFeatureExtractor):
                    The feature extractor used for extracting the features.
                examples (BatchEncoding):
                    The examples to be cleaned.
            Returns:
                BatchEncoding:
                    The cleaned examples.
            """
            # If there is more than on feature column list raise an exception
            if len(task_config.feature_column_names) != 1:
                raise ValueError(
                    "Only one feature column is supported, for the Automatic Speech Recognition task."
                )

            # Get the feature column name
            feature_column_name = task_config.feature_column_names[0]

            # Get the audio arrays
            audio_arrays = [x["array"] for x in examples[feature_column_name]]

            # Resample and pad the audio arrays
            inputs = feature_extractor(
                audio_arrays,
                sampling_rate=feature_extractor.sampling_rate,
                padding=True,
                max_length=feature_extractor.sampling_rate
                * 10,  # Allow 10 seconds audio
                truncation=True,
            )
            examples["input_features"] = inputs.data["input_features"]
            return examples

        # Preprocess the transcriptions
        examples["sentence"] = [
            clean_transcription(example)
            for example in examples[task_config.label_column_name]
        ]

        # Create labels column
        examples["labels"] = tokenizer.encode(
            list(examples[task_config.label_column_name])
        )

        # Load feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_config.model_id)

        # Clean the audio
        examples = clean_audio(feature_extractor, examples)

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
        self, processor: PreTrainedTokenizerBase
    ) -> DataCollatorCTCWithPadding:
        processor = Wav2Vec2Processor.from_pretrained("openai/whisper-small")
        return DataCollatorCTCWithPadding(processor=processor)

    def _prepare_predictions_and_labels(
        self,
        predictions: Sequence,
        dataset: Dataset,
        prepared_dataset: Dataset,
        **kwargs,
    ) -> List[Tuple[list, list]]:
        return [([], []), ([], [])]

    def _check_if_model_is_trained_for_task(self, model_predictions: list) -> bool:
        return True
