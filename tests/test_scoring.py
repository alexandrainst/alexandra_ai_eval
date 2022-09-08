"""Unit tests for the `scoring` module."""

import numpy as np
import pytest

from src.aiai_eval.scoring import aggregate_scores, log_scores


@pytest.fixture(scope="module")
def scores(metric_config):
    yield [
        {metric_config.name: 0.50},
        {metric_config.name: 0.55},
        {metric_config.name: 0.60},
    ]


class TestAggregateScores:
    def test_scores(self, scores, metric_config):
        # Aggregate scores using the `agg_scores` function
        agg_scores = aggregate_scores(scores=scores, metric_config=metric_config)

        # Manually compute the mean and standard error of the scores
        test_scores = [dct[metric_config.name] for dct in scores]
        mean = np.mean(test_scores)
        se = 1.96 * np.std(test_scores, ddof=1) / np.sqrt(len(test_scores))

        # Assert that `aggregate_scores` computed the same
        assert agg_scores == (mean, se)

    def test_no_scores(self, metric_config):
        agg_scores = aggregate_scores(scores=list(), metric_config=metric_config)
        for score in agg_scores:
            assert np.isnan(score)


class TestLogScores:
    @pytest.fixture(scope="class")
    def logged_scores(self, metric_config, scores):
        yield log_scores(
            task_name="task",
            metric_configs=[metric_config],
            scores=scores,
            model_id="model_id",
        )

    def test_is_correct_type(self, logged_scores):
        assert isinstance(logged_scores, dict)

    def test_has_correct_keys(self, logged_scores):
        assert sorted(logged_scores.keys()) == ["raw", "total"]

    def test_raw_scores_are_identical_to_input(self, logged_scores, scores):
        assert logged_scores["raw"] == scores

    def test_total_scores_is_dict(self, logged_scores):
        assert isinstance(logged_scores["total"], dict)

    def test_total_scores_keys(self, logged_scores, metric_config):
        assert sorted(logged_scores["total"].keys()) == [
            metric_config.name,
            f"{metric_config.name}_se",
        ]

    def test_total_scores_values_are_floats(self, logged_scores):
        for val in logged_scores["total"].values():
            assert isinstance(val, float)
