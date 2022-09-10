"""Class for the named entity recognition task."""

from copy import deepcopy
from functools import partial
from typing import List, Optional, Sequence, Tuple

import numpy as np
from datasets.arrow_dataset import Dataset
from spacy.tokens import Token
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorForTokenClassification,
)
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .config import TaskConfig
from .exceptions import InvalidTokenizer, MissingLabel
from .task import Task
from .utils import has_floats


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

    def _pytorch_preprocess_fn(
        self,
        examples: BatchEncoding,
        tokenizer: PreTrainedTokenizerBase,
        pytorch_model_config: PretrainedConfig,
        task_config: TaskConfig,
    ) -> BatchEncoding:
        return tokenize_and_align_labels(
            examples=examples,
            tokenizer=tokenizer,
            model_label2id=pytorch_model_config.label2id,
            dataset_id2label=task_config.id2label,
            label_column_name=task_config.label_column_name,
        )

    def _spacy_preprocess_fn(self, examples: BatchEncoding) -> BatchEncoding:
        examples["labels"] = [
            [self.task_config.id2label[ner_tag_id] for ner_tag_id in ner_tag_list]
            for ner_tag_list in examples["ner_tags"]
        ]
        return examples

    def _extract_spacy_predictions(self, tokens_processed: tuple) -> list:

        tokens, processed = tokens_processed

        # Get the aligned predictions
        aligned_spacy_tokens = align_spacy_tokens_with_gold_tokens(
            spacy_tokens=processed, gold_tokens=tokens
        )

        # Get the token labels
        get_ent_fn = partial(
            get_ent,
            dataset_id2label=self.task_config.id2label,
            dataset_label2id=self.task_config.label2id,
        )
        spacy_tags = list(map(get_ent_fn, processed))

        # Get the aligned labels
        aligned_spacy_predictions = [spacy_tags[i] for i in aligned_spacy_tokens]

        return aligned_spacy_predictions

    def _load_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> DataCollator:
        return DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    def _prepare_predictions_and_labels(
        self,
        predictions: Sequence,
        dataset: Dataset,
        prepared_dataset: Dataset,
        **kwargs,
    ) -> List[Tuple[list, list]]:

        # Extract the labels from the dataset
        labels = prepared_dataset["labels"]

        # Collapse the logits into single predictions for every sample
        if any(has_floats(pred) for pred in predictions):
            predictions = np.argmax(predictions, axis=-1)

        # Remove ignored index from predictions and labels
        predictions, labels = remove_ignored_index_from_predictions_and_labels(
            predictions=list(predictions),
            labels=labels,
            model_id2label=kwargs.get("model_id2label"),
            index_to_ignore=-100,
        )

        # Replace unknown tags present in the predictions to corresponding MISC tags
        predictions = replace_unknown_tags_with_misc_tags(
            list_of_tag_lists=list(predictions),
            dataset_id2label=self.task_config.id2label,
        )

        # Remove MISC tags from predictions and labels
        predictions_no_misc = remove_misc_tags(list_of_tag_lists=predictions)
        labels_no_misc = remove_misc_tags(list_of_tag_lists=labels)

        # Return the predictions and labels, both with and without MISC tags
        return [(predictions, labels), (predictions_no_misc, labels_no_misc)]

    def _check_if_model_is_trained_for_task(self, model_predictions: list) -> bool:

        sample_preds = model_predictions[0]
        has_sequence_elements = len(sample_preds[0]) > 0
        leaves_are_floats = isinstance(sample_preds[0][0], float)
        elements_are_strings = isinstance(sample_preds[0], str)

        return (has_sequence_elements and leaves_are_floats) or elements_are_strings


def tokenize_and_align_labels(
    examples: BatchEncoding,
    tokenizer: PreTrainedTokenizerBase,
    model_label2id: dict,
    dataset_id2label: list,
    label_column_name: str,
) -> BatchEncoding:
    """Tokenize all texts and align the labels with them.

    Args:
        examples (BatchEncoding):
            The examples to be tokenized.
        tokenizer (Hugging Face tokenizer):
            A pretrained tokenizer.
        model_label2id (dict):
            A dictionary that converts NER tags to IDs.
        dataset_id2label (list):
            A list that maps IDs to NER tags.
        label_column_name (str):
            The name of the label column.

    Returns:
        BatchEncoding:
            The tokenized data as well as labels.
    """

    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=True,
    )
    all_labels: List[List[int]] = []
    for i, ner_tags in enumerate(examples[label_column_name]):
        labels = [dataset_id2label[ner_tag] for ner_tag in ner_tags]
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
        label_ids: List[int] = list()
        for word_idx in word_ids:

            # Special tokens have a word id that is None. We set the label to -100
            # so they are automatically ignored in the loss function
            if word_idx is None:
                label_ids.append(-100)

            # We set the label for the first token of each word
            elif word_idx != previous_word_idx:
                label = labels[word_idx]
                try:
                    label_id = model_label2id[label.upper()]
                except KeyError:
                    raise MissingLabel(label=label, label2id=model_label2id)
                label_ids.append(label_id)

            # For the other tokens in a word, we set the label to -100
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def get_ent(token: Token, dataset_id2label: list, dataset_label2id: dict) -> str:
    """Extracts the entity from a SpaCy token.

    Args:
        token (spaCy Token):
            The inputted token from spaCy.
        dataset_id2label (list):
            A list that maps IDs to NER tags.
        dataset_label2id (dict):
            A dictionary that converts NER tags (and their synonyms) to IDs.

    Returns:
        str:
            The entity of the token.
    """

    # Deal with the O tag separately, as it is the only tag not of the form
    # B-tag or I-tag
    if token.ent_iob_ == "O":
        return "O"

    # Otherwise the tag is of the form B-tag or I-tag for some NER tag
    else:
        # Extract tag from spaCy token
        ent = f"{token.ent_iob_}-{token.ent_type_}"

        # Get the ID of the MISC tag, which we will use as a backup, in case the
        # given tag is not in the dataset
        misc_idx = dataset_label2id[f"{token.ent_iob_}-MISC".upper()]

        # Convert the tag to the its canonical synonym, or to the MISC tag if it
        # is not in the dataset
        return dataset_id2label[dataset_label2id.get(ent, misc_idx)]


def remove_ignored_index_from_predictions_and_labels(
    predictions: List[list],
    labels: List[list],
    model_id2label: Optional[List[str]],
    index_to_ignore: int = -100,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Removes the ignored index from the predictions and labels.

    Args:
        predictions (list of lists):
            The predicted labels.
        labels (list of lists):
            The true labels.
        model_id2label (list of str, or None):
            A list that maps IDs to NER tags. If None then the predictions and labels
            will not be modified.
        index_to_ignore (int, optional):
            The index to ignore. Defaults to -100.

    Returns:
        tuple of list of list of str:
            The predictions and labels with the ignored index removed.
    """
    # If `model_id2label` is None then we simply return the predictions and labels
    if model_id2label is None:
        return predictions, labels

    # Otherwise, we firstly remove the ignored index from the predictions, using the
    # labels
    predictions = [
        [
            model_id2label[pred_id]
            for pred_id, lbl_id in zip(pred, label)
            if lbl_id != index_to_ignore
        ]
        for pred, label in zip(predictions, labels)
    ]

    # Next, we remove the ignored index from the labels
    labels = [
        [model_id2label[lbl_id] for lbl_id in label if lbl_id != index_to_ignore]
        for label in labels
    ]

    # Finally, we return the predictions and labels
    return predictions, labels


def replace_unknown_tags_with_misc_tags(
    list_of_tag_lists: List[List[str]],
    dataset_id2label: List[str],
) -> List[List[str]]:
    """Replaces unknown tags with MISC tags.

    This replaces the predicted tags with either MISC or O tags if they are not part of
    the dataset. We use the `id2label` from the dataset here, as opposed to the model's
    `id2label` mapping, since we want to replace all the tags which do not appear in
    the *dataset labels* with either MISC or O tags.

    Args:
        list_of_tag_lists (list of list of str):
            A list of lists containing NER tags.
        dataset_id2label (list of str):
            The mapping from label IDs to labels.

    Returns:
        list of list of str:
            The list of lists containing NER tags with unknown tags replaced with MISC
            tags.
    """
    # Use the `id2label` mapping to get a list of all the non-MISC NER tags present in
    # the dataset
    dataset_labels_without_misc = set(dataset_id2label).difference({"B-MISC", "I-MISC"})

    # Iterate over the nested tags, and replace them with MISC tags if they are not
    # present in the dataset
    for i, tag_list in enumerate(list_of_tag_lists):
        for j, ner_tag in enumerate(tag_list):
            if ner_tag not in dataset_labels_without_misc:
                if ner_tag[:2] == "B-":
                    list_of_tag_lists[i][j] = "B-MISC"
                elif ner_tag[:2] == "I-":
                    list_of_tag_lists[i][j] = "I-MISC"
                else:
                    list_of_tag_lists[i][j] = "O"

    # Return the list of lists containing NER tags with unknown tags replaced with MISC
    # tags
    return list_of_tag_lists


def remove_misc_tags(list_of_tag_lists: List[List[str]]) -> List[List[str]]:
    """Removes MISC tags from a list of lists of tags.

    Args:
        list_of_tag_lists (list of list of str):
            A list of lists containing NER tags.

    Returns:
        list of list of str:
            The list of lists containing NER tags with MISC tags removed.
    """
    # Make a copy of the list, to ensure that we don't get any side effects
    list_of_tag_lists = deepcopy(list_of_tag_lists)

    # Iterate over the nested tags, and remove them if they are MISC tags
    for i, tag_list in enumerate(list_of_tag_lists):
        for j, ner_tag in enumerate(tag_list):
            if ner_tag == "B-MISC" or ner_tag == "I-MISC":
                list_of_tag_lists[i][j] = "O"

    # Return the list of lists containing NER tags with MISC tags removed
    return list_of_tag_lists


def align_spacy_tokens_with_gold_tokens(
    spacy_tokens: List[Token],
    gold_tokens: List[str],
) -> List[int]:
    """Aligns spaCy tokens with gold tokens.

    This is necessary because spaCy's tokenizer is different to the tokenizer used by
    the dataset. This function aligns the tokens by inserting empty tokens where
    necessary.

    Args:
        spacy_tokens (list of Token):
            A list of spaCy tokens.
        gold_tokens (list of str):
            A list of gold tokens.

    Returns:
        list of int:
            A list of indices of `spacy_tokens` that correspond to the gold tokens.
    """
    # Get the alignment between the SpaCy model's tokens and the gold tokens
    gold_token_idxs = [
        tok_idx for tok_idx, tok in enumerate(gold_tokens) for _ in str(tok)
    ]
    spacy_token_idxs = [
        tok_idx for tok_idx, tok in enumerate(spacy_tokens) for _ in str(tok)
    ]
    alignment = list(zip(gold_token_idxs, spacy_token_idxs))

    # Get the aligned predictions
    predictions: List[int] = list()
    for idx, _ in enumerate(gold_tokens):
        aligned_pred_token = [
            spacy_token_idx
            for gold_token_idx, spacy_token_idx in alignment
            if gold_token_idx == idx
        ][0]
        predictions.append(aligned_pred_token)

    # Return the aligned predictions
    return predictions
