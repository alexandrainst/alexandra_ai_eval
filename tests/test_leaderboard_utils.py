"""Unit tests for the `leaderboard_utils` module."""

import pandas as pd
import pytest
from requests.exceptions import ConnectionError

from alexandra_ai_eval.leaderboard_utils import Session


class TestSession:
    @pytest.fixture(scope="class")
    def base_url(self):
        return "https://api.aiai.alexandrainst.dk"

    @pytest.fixture(scope="class")
    def session(self, base_url):
        return Session(base_url=base_url)

    def test_get_task(self, session):
        task = session.get_task("sentiment-classification")
        assert isinstance(task, dict)

    def test_get_task_raw(self, session):
        task = session.get_task("sentiment-classification", raw=True)
        assert isinstance(task, dict)

    def test_get_task_raises_error_if_task_not_found(self, session):
        with pytest.raises(ValueError):
            session.get_task("invalid-task")

    def test_get_model_for_task_raw(self, session):
        task_dict = session.get_task("sentiment-classification", raw=True)
        if len(task_dict["id"]) > 0:
            task = pd.DataFrame.from_dict(task_dict)
            existing_model = task["model_id"].values[0]
            model = session.get_model_for_task(
                "sentiment-classification",
                existing_model,
                raw=True,
            )
            assert model["name"] == existing_model
        else:
            pytest.skip("No models for task")

    def test_get_model_for_task(self, session):
        task_dict = session.get_task("sentiment-classification")
        if len(task_dict["id"]) > 0:
            task = pd.DataFrame.from_dict(task_dict)
            existing_model = task["model_id"].values[0]
            model = session.get_model_for_task(
                "sentiment-classification",
                existing_model,
            )
            assert model["name"] == existing_model
        else:
            pytest.skip("No models for task")

    def test_get_model_for_task_raises_error_if_model_not_found(self, session):
        with pytest.raises(ValueError):
            session.get_model_for_task("sentiment-classification", "invalid-model")

    def test_get_model_for_task_raises_error_if_model_not_found_raw(self, session):
        with pytest.raises(ValueError):
            session.get_model_for_task(
                "sentiment-classification", "invalid-model", raw=True
            )

    def test_post_model_to_task(self, session):
        response = session.post_model_to_task(
            model_type="huggingface",
            task_name="sentiment-classification",
            model_id="pin/senda",
            metrics={"mcc": 0.5, "mcc_se": 0.1, "macro_f1": 0.5, "macro_f1_se": 0.1},
            test=True,
        )
        assert isinstance(response, dict)

    def test_post_model_to_task_raises_error_if_task_not_found(self, session):
        with pytest.raises(ValueError):
            session.post_model_to_task(
                model_type="huggingface",
                task_name="invalid-task",
                model_id="pin/senda",
                metrics={
                    "mcc": 0.5,
                    "mcc_se": 0.1,
                    "macro_f1": 0.5,
                    "macro_f1_se": 0.1,
                },
                test=True,
            )

    def test_post_model_to_task_gives_error_if_model_type_not_found(self, session):
        response = session.post_model_to_task(
            model_type="invalid-model-type",
            task_name="sentiment-classification",
            model_id="pin/senda",
            metrics={
                "mcc": 0.5,
                "mcc_se": 0.1,
                "macro_f1": 0.5,
                "macro_f1_se": 0.1,
            },
            test=True,
        )
        assert response["error"] == "Model type not found"

    def test_post_model_to_task_gives_error_if_metrics_not_found(self, session):
        response = session.post_model_to_task(
            model_type="huggingface",
            task_name="sentiment-classification",
            model_id="pin/senda",
            metrics={"invalid-metric": 0.5},
            test=True,
        )
        assert response["error"] == "One of more metrics not found"

    def test_post_model_to_task_raises_error_if_wrong_session_url(self):
        with pytest.raises(ConnectionError):
            session = Session(base_url="https://invalid.url.com/invalid")
            session.post_model_to_task(
                model_type="huggingface",
                task_name="sentiment-classification",
                model_id="pin/senda",
                metrics={
                    "mcc": 0.5,
                    "mcc_se": 0.1,
                    "macro_f1": 0.5,
                    "macro_f1_se": 0.1,
                },
                test=True,
            )
