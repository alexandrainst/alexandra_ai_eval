"""Unit tests for the `task_configs` module."""

import pytest

from alexandra_ai_eval.config import TaskConfig
from alexandra_ai_eval.task_configs import get_all_task_configs


class TestGetAllTaskConfigs:
    @pytest.fixture(scope="class")
    def task_configs(self):
        yield get_all_task_configs()

    def test_task_configs_is_dict(self, task_configs):
        assert isinstance(task_configs, dict)

    def test_task_configs_are_objects(self, task_configs):
        for task_config in task_configs.values():
            assert isinstance(task_config, TaskConfig)
