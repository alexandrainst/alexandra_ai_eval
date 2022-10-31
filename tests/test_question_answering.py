"""Unit tests for the `question_answering` module."""

from collections import defaultdict
from functools import partial

import numpy as np
import pytest

from aiai_eval.enums import Framework
from aiai_eval.model_loading import get_model_config, load_model
from aiai_eval.question_answering import (
    QuestionAnswering,
    find_best_answer,
    find_valid_answers,
    postprocess_labels,
    postprocess_predictions,
    prepare_test_examples,
)
from aiai_eval.task_configs import QA


@pytest.fixture(scope="module")
def qa(evaluation_config):
    yield QuestionAnswering(task_config=QA, evaluation_config=evaluation_config)


@pytest.fixture(scope="module")
def dataset(qa):
    full_dataset = qa._load_data()
    yield full_dataset.select(range(20))


@pytest.fixture(scope="module")
def model_dict(evaluation_config):
    model_config = get_model_config(
        model_id="saattrupdan/electra-small-qa-da",
        task_config=QA,
        evaluation_config=evaluation_config,
    )
    yield load_model(
        model_config=model_config,
        task_config=QA,
        evaluation_config=evaluation_config,
    )


@pytest.fixture(scope="class")
def cls_token_index(model_dict):
    yield model_dict["tokenizer"].cls_token_id


@pytest.fixture(scope="module")
def prepared_dataset(dataset, model_dict):
    map_fn = partial(prepare_test_examples, tokenizer=model_dict["tokenizer"])
    yield dataset.map(map_fn, batched=True, remove_columns=dataset.column_names)


@pytest.fixture(scope="module")
def features_per_example(dataset, prepared_dataset):
    id_to_index = {k: i for i, k in enumerate(dataset["id"])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(prepared_dataset):
        id = feature["id"]
        example_index = id_to_index[id]
        features_per_example[example_index].append(i)
    yield features_per_example


@pytest.fixture(
    scope="class",
    params=[True, False],
    ids=["list-predictions", "array-predictions"],
)
def predictions(qa, model_dict, request, prepared_dataset):
    list_predictions = qa._get_model_predictions(
        model=model_dict["model"],
        tokenizer=model_dict["tokenizer"],
        processor=model_dict["processor"],
        prepared_dataset=prepared_dataset,
        batch_size=2,
        framework=Framework.PYTORCH,
    )
    if request.param:
        return list_predictions
    else:
        return np.array(list_predictions)


class TestPrepareTestExamples:
    def test_prepared_dataset_has_input_ids(self, prepared_dataset):
        assert "input_ids" in prepared_dataset.features

    def test_prepared_dataset_has_attention_mask(self, prepared_dataset):
        assert "attention_mask" in prepared_dataset.features

    def test_prepared_dataset_has_id_column(self, prepared_dataset):
        assert "id" in prepared_dataset.features

    def test_prepared_dataset_has_offset_mapping(self, prepared_dataset):
        assert "offset_mapping" in prepared_dataset.features


class TestPostprocessPredictions:
    @pytest.fixture(scope="class")
    def postprocessed_predictions(
        self, predictions, dataset, prepared_dataset, cls_token_index
    ):
        yield postprocess_predictions(
            predictions=predictions,
            dataset=dataset,
            prepared_dataset=prepared_dataset,
            cls_token_index=cls_token_index,
        )

    def test_output_is_list(self, postprocessed_predictions):
        assert isinstance(postprocessed_predictions, list)

    def test_output_entries_are_dicts(self, postprocessed_predictions):
        for entry in postprocessed_predictions:
            assert isinstance(entry, dict)

    def test_output_entry_keys(self, postprocessed_predictions):
        for entry in postprocessed_predictions:
            assert set(entry.keys()) == {
                "id",
                "prediction_text",
                "no_answer_probability",
            }

    def test_no_answer_probability_is_always_zero(self, postprocessed_predictions):
        for entry in postprocessed_predictions:
            assert entry["no_answer_probability"] == 0.0


class TestFindBestAnswer:
    @pytest.fixture(scope="class")
    def all_start_logits(self, predictions):
        yield np.asarray(predictions)[:, :, 0]

    @pytest.fixture(scope="class")
    def all_end_logits(self, predictions):
        yield np.asarray(predictions)[:, :, 1]

    @pytest.mark.parametrize(
        argnames=[
            "example_index",
            "max_answer_length",
            "num_best_logits",
            "min_null_score",
            "expected_answer",
        ],
        argvalues=[
            (
                5,
                30,
                20,
                0.0,
                "1796 til 1811 14 budgetoverskud og 2 underskud. Der var "
                "en kraftig stigning i gælden som følge af krigen i 1812",
            ),
            (11, 30, 20, 0.0, "Penny Marshall"),
            (5, 5, 20, 0.0, "1796"),
            (11, 30, 1, 0.0, "Penny Marshall og har Geena Davis, Tom Hanks"),
            (11, 30, 20, 10.0, ""),
            (7, 30, 20, 0.0, ""),
        ],
        ids=[
            "example1",
            "example2",
            "example1-with-short-answer-length",
            "example2-with-few-best-logits",
            "example2-with-high-min-null-score",
            "non-existing-answer",
        ],
    )
    def test_find_best_answer(
        self,
        example_index,
        max_answer_length,
        num_best_logits,
        min_null_score,
        expected_answer,
        features_per_example,
        all_start_logits,
        all_end_logits,
        dataset,
        prepared_dataset,
        cls_token_index,
    ):
        best_answer = find_best_answer(
            all_start_logits=all_start_logits,
            all_end_logits=all_end_logits,
            prepared_dataset=prepared_dataset,
            feature_indices=features_per_example[example_index],
            context=dataset[example_index]["context"],
            cls_token_index=cls_token_index,
            max_answer_length=max_answer_length,
            num_best_logits=num_best_logits,
            min_null_score=min_null_score,
        )
        assert best_answer == expected_answer


class TestFindValidAnswers:
    @pytest.fixture(scope="class")
    def example_index(self):
        yield 11

    @pytest.fixture(scope="class")
    def feature_index(self, example_index, features_per_example):
        yield features_per_example[example_index][0]

    @pytest.fixture(scope="class")
    def start_logits(self, predictions, feature_index):
        yield np.asarray(predictions)[feature_index, :, 0]

    @pytest.fixture(scope="class")
    def end_logits(self, predictions, feature_index):
        yield np.asarray(predictions)[feature_index, :, 1]

    @pytest.fixture(scope="class")
    def offset_mapping(self, prepared_dataset, feature_index):
        yield prepared_dataset[feature_index]["offset_mapping"]

    @pytest.fixture(scope="class")
    def context(self, dataset, example_index):
        yield dataset[example_index]["context"]

    @pytest.mark.parametrize(
        argnames=[
            "max_answer_length",
            "num_best_logits",
            "min_null_score",
            "expected_answers",
        ],
        argvalues=[
            (
                30,
                3,
                0.0,
                [
                    {
                        "score": 7.4,
                        "text": "Penny Marshall og har Geena Davis, Tom Hanks",
                    },
                    {
                        "score": 7.1,
                        "text": "Penny Marshall og har Geena Davis, Tom Hanks, Madonna "
                        "og Lori Petty",
                    },
                    {
                        "score": 7.0,
                        "text": "Penny Marshall og har Geena Davis",
                    },
                    {
                        "score": 7.0,
                        "text": "Lori Petty",
                    },
                    {
                        "score": 6.9,
                        "text": "Madonna og Lori Petty",
                    },
                ],
            ),
            (
                4,
                3,
                0.0,
                [
                    {
                        "score": 7.0,
                        "text": "Lori Petty",
                    },
                ],
            ),
            (
                30,
                1,
                0.0,
                [
                    {
                        "score": 7.4,
                        "text": "Penny Marshall og har Geena Davis, Tom Hanks",
                    }
                ],
            ),
            (
                30,
                3,
                7.0,
                [
                    {
                        "score": 7.4,
                        "text": "Penny Marshall og har Geena Davis, Tom Hanks",
                    },
                    {
                        "score": 7.1,
                        "text": "Penny Marshall og har Geena Davis, Tom Hanks, Madonna "
                        "og Lori Petty",
                    },
                    {
                        "score": 7.0,
                        "text": "Lori Petty",
                    },
                ],
            ),
        ],
        ids=[
            "default",
            "short-answer-length",
            "few-best-logits",
            "high-min-null-score",
        ],
    )
    def test_find_valid_answers(
        self,
        max_answer_length,
        num_best_logits,
        min_null_score,
        expected_answers,
        start_logits,
        end_logits,
        offset_mapping,
        context,
    ):
        valid_answers = find_valid_answers(
            start_logits=start_logits,
            end_logits=end_logits,
            offset_mapping=offset_mapping,
            context=context,
            max_answer_length=max_answer_length,
            num_best_logits=num_best_logits,
            min_null_score=min_null_score,
        )
        valid_answers = [
            dict(score=round(a["score"], 1), text=a["text"]) for a in valid_answers
        ]
        assert valid_answers == expected_answers


def test_postprocess_labels(dataset):
    truncated_dataset = dataset.select(range(1))
    postprocessed_labels = postprocess_labels(truncated_dataset)
    assert postprocessed_labels == [
        {
            "answers": {"answer_start": [9700], "text": ["2013"]},
            "id": 7960403399608384663,
        },
    ]
