"""Class for the named entity recognition task."""

from copy import deepcopy
from functools import partial
from typing import List, Sequence, Tuple

import numpy as np
from datasets.arrow_dataset import Dataset
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

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

    def _preprocess_data(self, dataset: Dataset, framework: str, **kwargs) -> Dataset:
        """Preprocess the data.

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
                "Evaluation of named entity recognition for SpaCy models is not yet "
                "implemented."
            )

        # We are now assuming we are using pytorch
        map_fn = partial(
            self._tokenize_and_align_labels,
            tokenizer=kwargs["tokenizer"],
            label2id=kwargs["config"].label2id,
        )
        tokenised_dataset = dataset.map(
            map_fn,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return tokenised_dataset

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
        for i, ner_tags in enumerate(examples[self.task_config.label_column_name]):
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
                        tokenizer_type=tokenizer_type,
                        message=(
                            "The tokens could not be aligned with the words during "
                            "manual word-token alignment. It seems that the tokenizer "
                            "is neither of the fast variant nor of a SentencePiece/"
                            "WordPiece variant. The tokenizer type is "
                            f"{tokenizer_type}."
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

    def _load_data_collator(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> DataCollatorForTokenClassification:
        """Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (Hugging Face tokenizer):
                A pretrained tokenizer.

        Returns:
            Hugging Face data collator:
                The data collator.
        """
        return DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

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
                The predictions of the model, which can be either class labels or
                logits.
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
        # Extract the `model_id2label` mapping
        model_id2label = kwargs["model_id2label"]

        # Extract the labels from the dataset
        labels = prepared_dataset["labels"]

        # Collapse the logits into single predictions for every sample
        if any(
            np.asarray(predictions).dtype == dtype
            for dtype in {np.float16, np.float32, np.float64}
        ):
            predictions = np.argmax(predictions, axis=-1)

        # Remove ignored index from predictions and labels
        if model_id2label is not None:
            predictions = [
                [
                    model_id2label[pred_id]
                    for pred_id, lbl_id in zip(pred, label)
                    if lbl_id != -100
                ]
                for pred, label in zip(predictions, labels)
            ]
            labels = [
                [model_id2label[lbl_id] for lbl_id in label if lbl_id != -100]
                for label in labels
            ]

        # Replace predicted tag with either MISC or O tags if they are not part of the
        # dataset. We use the `id2label` from the dataset here, as opposed to the above
        # `model_id2label`, since we want to replace all the tags which do not appear
        # in the *dataset labels* with either MISC or O tags.
        dataset_labels_without_misc = set(self.task_config.id2label).difference(
            {"B-MISC", "I-MISC"}
        )
        for i, prediction_list in enumerate(predictions):
            for j, ner_tag in enumerate(prediction_list):
                if ner_tag not in dataset_labels_without_misc:
                    if ner_tag[:2] == "B-":
                        predictions[i][j] = "B-MISC"
                    elif ner_tag[:2] == "I-":
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

        return [
            (list(predictions), labels),
            (list(predictions_no_misc), labels_no_misc),
        ]
