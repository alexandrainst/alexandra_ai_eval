"""Class for the named entity recognition task."""

from copy import deepcopy
from functools import partial
from typing import List, Sequence, Tuple

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from tqdm import tqdm
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .exceptions import InvalidEvaluation, InvalidTokenizer, MissingLabel
from .task import Task


def tokenize_and_align_labels(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    model_label2id: dict,
    dataset_id2label: list,
    label_column_name: str,
) -> BatchEncoding:
    """Tokenise all texts and align the labels with them.

    Args:
        examples (dict):
            The examples to be tokenised.
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

        if framework == "spacy":
            raise InvalidEvaluation(
                "Evaluation of named entity recognition for SpaCy models is not yet "
                "implemented."
            )

        # We are now assuming we are using pytorch
        map_fn = partial(
            tokenize_and_align_labels,
            tokenizer=kwargs["tokenizer"],
            model_label2id=kwargs["model_config"].label2id,
            dataset_id2label=self.task_config.id2label,
            label_column_name=self.task_config.label_column_name,
        )
        tokenised_dataset = dataset.map(
            map_fn,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return tokenised_dataset

    def _preprocess_data_spacy(self, dataset: Dataset) -> Dataset:
        """Preprocess the given Huggingface dataset for use by a SpaCy model.

        Args:
            dataset (Dataset): The dataset to preprocess.

        Returns:
            Dataset:
                The preprocessed dataset.
        """
        # Add a labels column to the dataset
        def create_label_col(example):
            example["labels"] = [
                self.task_config.id2label[x] for x in example["labels"]
            ]
            return example

        dataset = dataset.add_column("labels", dataset["ner_tags"])
        return dataset.map(create_label_col)

    def _get_spacy_predictions_and_labels(
        self, model, dataset: Dataset, batch_size: int
    ) -> tuple:
        """Get predictions from SpaCy model on dataset.

        Args:
            model (SpaCy model):
                The model.
            dataset (Hugging Face dataset):
                The dataset.
            batch_size (int):
                The batch size to use.

        Returns:
            A pair of arrays:
                The first array contains the probability predictions and the second
                array contains the true labels.
        """
        # Initialise progress bar
        if self.evaluation_config.progress_bar:
            itr = tqdm(
                dataset[self.task_config.feature_column_names[0]],
                desc="Evaluating model",
                leave=False,
            )
        else:
            itr = dataset[self.task_config.feature_column_names[0]]

        processed = model.pipe(itr, batch_size=batch_size)
        map_fn = self._extract_spacy_predictions
        predictions = map(map_fn, zip(dataset["tokens"], processed))

        return list(predictions), dataset["labels"]

    def _extract_spacy_predictions(self, tokens_processed: tuple) -> list:
        """Helper function that extracts the predictions from a SpaCy model.

        Aside from extracting the predictions from the model, it also aligns the
        predictions with the gold tokens, in case the SpaCy tokeniser tokenises the
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
        tokens, processed = tokens_processed

        # Get the token labels
        token_labels = self._get_spacy_token_labels(processed)

        # Get the alignment between the SpaCy model's tokens and the gold tokens
        token_idxs = [tok_idx for tok_idx, tok in enumerate(tokens) for _ in str(tok)]
        pred_token_idxs = [
            tok_idx for tok_idx, tok in enumerate(processed) for _ in str(tok)
        ]
        alignment = list(zip(token_idxs, pred_token_idxs))

        # Get the aligned predictions
        predictions = list()
        for tok_idx, _ in enumerate(tokens):
            aligned_pred_token = [
                pred_token_idx
                for token_idx, pred_token_idx in alignment
                if token_idx == tok_idx
            ][0]
            predictions.append(token_labels[aligned_pred_token])

        return predictions

    def _get_spacy_token_labels(self, processed) -> Sequence[str]:
        """Get predictions from SpaCy model on dataset.

        Args:
            model (SpaCy model):
                The model.
            dataset (Hugging Face dataset):
                The dataset.

        Returns:
            A list of strings:
                The predicted NER labels.
        """

        def get_ent(token) -> str:
            """Helper function that extracts the entity from a SpaCy token.

            Args:
                token (spaCy Token):
                    The inputted token from spaCy.

            Returns:
                str:
                    The entity of the token.
            """

            # Deal with the O tag separately, as it is the only tag not of the form
            # B-tag or I-tag
            if token.ent_iob_ == "O":
                return "O"

            # In general return a tag of the form B-tag or I-tag
            else:
                # Extract tag from spaCy token
                ent = f"{token.ent_iob_}-{token.ent_type_}"

                # Convert the tag to the its canonical synonym
                alt_idx = self.task_config.label2id[f"{token.ent_iob_}-MISC".upper()]
                return self.task_config.id2label[
                    self.task_config.label2id.get(ent, alt_idx)
                ]

        return [get_ent(token) for token in processed]

    def _load_data_collator(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> DataCollatorForTokenClassification:

        return DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    def _prepare_predictions_and_labels(
        self,
        predictions: Sequence,
        dataset: Dataset,
        prepared_dataset: Dataset,
        **kwargs,
    ) -> List[Tuple[list, list]]:

        # Extract the `model_id2label` mapping
        model_id2label = kwargs["model_id2label"]

        # Extract the labels from the dataset
        labels = prepared_dataset["labels"]

        # Collapse the logits into single predictions for every sample
        if any(
            np.asarray(pred).dtype == dtype
            for dtype in {np.float16, np.float32, np.float64}
            for pred in predictions
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

    def _check_if_model_is_trained_for_task(self, model_predictions: list) -> bool:
        """Check if the model is trained for the task.

        Args:
            model_predictions (list):
                The predictions of the model.

        Returns:
            bool:
                True if the model is trained for the task, False otherwise.
        """
        sample_preds = model_predictions[0]

        # Check if output comes from a pytorch or spacy model.
        if isinstance(sample_preds, torch.Tensor):
            try:
                if (
                    isinstance(sample_preds[0], torch.Tensor)
                    and len(sample_preds[0]) > 0
                    and isinstance(sample_preds[0][0].item(), float)
                ):
                    return True
                else:
                    return False
            except TypeError:
                # This happens if the output is a torch.Tensor with no length, i.e. the
                # model output fits sequence classification and not token classification.
                return False
        else:
            return isinstance(sample_preds, list) and isinstance(sample_preds[0], str)
