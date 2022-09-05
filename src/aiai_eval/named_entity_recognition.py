"""Class for the named entity recognition task."""
from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerBase

from .exceptions import InvalidEvaluation, InvalidTokenizer, MissingLabel
from .task import Task


class NamedEntityRecognition(Task):
    """Named entity recognition task.

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

    def _preprocess_data_transformer(
        self, dataset: Dataset, framework: str, **kwargs
    ) -> Dataset:
        """Preprocess a dataset by tokenizing and aligning the labels.

        For use by a transformer model.

        Args:
            dataset (Hugging Face dataset):
                The dataset to preprocess.
            framework (str):
                Specification of which framework the model is created in.
            kwargs:
                Extra keyword arguments containing objects used in preprocessing the
                dataset.

        Returns:
            Hugging Face dataset:
                The preprocessed dataset.
        """
        if framework == "spacy":
            raise InvalidEvaluation(
                "Evaluation of text predictions for SpaCy models is not yet "
                "implemented."
            )

        # Check what labels are present in the dataset, and store if MISC tags are not
        # present
        labels_in_train = {tag for tag_list in dataset["ner_tags"] for tag in tag_list}
        self.has_misc_tags = "B-MISC" in labels_in_train or "I-MISC" in labels_in_train

        # We are now assuming we are using pytorch
        map_fn = partial(
            self._tokenize_and_align_labels,
            tokenizer=kwargs["tokenizer"],
            label2id=kwargs["config"].label2id,
        )
        tokenised_dataset = dataset.map(
            map_fn, batched=True, load_from_cache_file=False
        )

        # Remove unused columns
        return tokenised_dataset.remove_columns(
            [
                self.task_config.feature_column_name,
                "tokens",
                "lemmas",
                "sent_id",
                "tok_ids",
                "pos_tags",
                "morph_tags",
                "dep_ids",
                "dep_labels",
                "ner_tags",
            ]
        )

    def _tokenize_and_align_labels(
        self, examples: dict, tokenizer, label2id: dict
    ) -> dict:
        """Tokenise all texts and align the labels with them.
        
        Args:
            examples (dict):
                The examples to be tokenised.
            tokenizer (Hugging Face tokenizer):
                A pretrained tokenizer.
            label2id (dict):
                A dictionary that converts NER tags to IDs.
                
        Returns:
            dict:
                A dictionary containing the tokenized data as well as labels.
        """

        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=True,
        )
        all_labels: List[List[int]] = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            labels = [self.task_config.id2label[ner_tag] for ner_tag in ner_tags]
            try:
                word_ids = tokenized_inputs.word_ids(batch_index=i)

            # This happens if the tokenizer is not of the fast variant, in which case
            # the `word_ids` method is not available, so we have to extract this
            # manually. It's slower, but it works, and it should only occur rarely,
            # when the Hugging Face team has not implemented a fast variant of the
            # tokenizer yet.
            except ValueError:

                # Get the list of words in the document
                words = examples["tokens"][i]

                # Get the list of token IDs in the document
                tok_ids = tokenized_inputs.input_ids[i]

                # Decode the token IDs
                tokens = tokenizer.convert_ids_to_tokens(tok_ids)

                # Remove prefixes from the tokens
                prefixes_to_remove = ["▁", "##"]
                for tok_idx, tok in enumerate(tokens):
                    for prefix in prefixes_to_remove:
                        tok = tok.lstrip(prefix)
                    tokens[tok_idx] = tok

                # Replace special tokens with `None`
                sp_toks = tokenizer.special_tokens_map.values()
                tokens = [None if tok in sp_toks else tok for tok in tokens]

                # Get the alignment between the words and the tokens, on a character
                # level
                word_idxs = [
                    word_idx for word_idx, word in enumerate(words) for _ in str(word)
                ]
                token_idxs = [
                    tok_idx
                    for tok_idx, tok in enumerate(tokens)
                    for _ in str(tok)
                    if tok is not None
                ]
                alignment = list(zip(word_idxs, token_idxs))

                # Raise error if there are not as many characters in the words as in
                # the tokens. This can be due to the use of a different prefix.
                if len(word_idxs) != len(token_idxs):
                    tokenizer_type = type(tokenizer).__name__
                    raise InvalidTokenizer(
                        tokenizer_type,
                        (
                            "The tokens could not be aligned with the words during manual "
                            "word-token alignment. It seems that the tokenizer is neither "
                            "of the fast variant nor of a SentencePiece/WordPiece variant. The "
                            f"tokenizer type is {tokenizer_type}."
                        ),
                    )

                # Get the aligned word IDs
                word_ids = list()
                for tok_idx, tok in enumerate(tokens):
                    if tok is None or tok == "":
                        word_ids.append(None)
                    else:
                        word_idx = [
                            word_idx
                            for word_idx, token_idx in alignment
                            if token_idx == tok_idx
                        ][0]
                        word_ids.append(word_idx)

            previous_word_idx = None
            label_ids: List[int] = []
            for word_idx in word_ids:

                # Special tokens have a word id that is None. We set the label to -100
                # so they are automatically ignored in the loss function
                if word_idx is None:
                    label_ids.append(-100)

                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label = labels[word_idx]
                    try:
                        label_id = label2id[label.upper()]
                    except KeyError:
                        raise MissingLabel(label=label, label2id=label2id)
                    label_ids.append(label_id)

                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(-100)

                previous_word_idx = word_idx

            all_labels.append(label_ids)
        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

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
        return DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    def _prepare_predictions_and_labels(
        self,
        predictions_np: np.ndarray,
        labels_np: np.ndarray,
        id2label: Optional[list] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Prepare predictions and labels for output.

        Args:
            predictions_np (NumPy array):
                The predictions of the model.
            labels_np (NumPy array):
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

        if id2label is not None:
            # Remove ignored index (special tokens)
            predictions = [
                [
                    id2label[pred_id]
                    for pred_id, lbl_id in zip(pred, label)
                    if lbl_id != -100
                ]
                for pred, label in zip(predictions_np, labels_np)
            ]
            labels = [
                [id2label[lbl_id] for _, lbl_id in zip(pred, label) if lbl_id != -100]
                for pred, label in zip(predictions_np, labels_np)
            ]

        # Replace predicted tag with either MISC or O tags if they are not part of the
        # dataset
        id2label_without_misc = set(self.task_config.id2label).difference(
            {"B-MISC", "I-MISC"}
        )
        for i, prediction_list in enumerate(predictions):
            for j, ner_tag in enumerate(prediction_list):
                if ner_tag not in id2label_without_misc:
                    if self.has_misc_tags and id2label[ner_tag][:2] == "B-":  # type: ignore
                        predictions[i][j] = "B-MISC"
                    elif self.has_misc_tags and id2label[ner_tag][:2] == "I-":  # type: ignore
                        predictions[i][j] = "I-MISC"
                    else:
                        predictions[i][j] = "O"

        # Remove MISC labels from predictions
        predictions_no_misc = deepcopy(predictions)
        for i, prediction_list in enumerate(predictions_no_misc):
            for j, ner_tag in enumerate(prediction_list):
                if ner_tag[-4:] == "MISC":
                    predictions_no_misc[i][j] = "O"

        # Remove MISC labels from labels
        labels_no_misc = deepcopy(labels)
        for i, label_list in enumerate(labels_no_misc):
            for j, ner_tag in enumerate(label_list):
                if ner_tag[-4:] == "MISC":
                    labels_no_misc[i][j] = "O"

        return [(predictions, labels), (predictions_no_misc, labels_no_misc)]

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
        full_preprocessed = self._preprocess_data_transformer(
            dataset=dataset, framework="pytorch", **kwargs
        )
        return full_preprocessed["input_ids"]
