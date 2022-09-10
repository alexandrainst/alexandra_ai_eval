"""Unit tests for the `co2` module."""

import pytest
from codecarbon import EmissionsTracker, OfflineEmissionsTracker

from aiai_eval.co2 import get_carbon_tracker
from aiai_eval.enums import CountryCode


class TestGetCarbonTracker:
    @pytest.fixture(scope="class")
    def params(self):
        yield dict(task_name="test_task", country_code=CountryCode.DNK, verbose=False)

    @pytest.fixture(scope="class")
    def online_carbon_tracker(self, params):
        yield get_carbon_tracker(**params, prefer_offline=False)

    @pytest.fixture(scope="class")
    def offline_carbon_tracker(self, params):
        yield get_carbon_tracker(**params, prefer_offline=True)

    def test_get_carbon_tracker_online(self, online_carbon_tracker):
        assert isinstance(online_carbon_tracker, EmissionsTracker)

    def test_get_carbon_tracker_offline(self, offline_carbon_tracker):
        assert isinstance(offline_carbon_tracker, OfflineEmissionsTracker)

    def test_project_name(self, online_carbon_tracker, offline_carbon_tracker, params):
        assert online_carbon_tracker._conf["project_name"] == params["task_name"]
        assert offline_carbon_tracker._conf["project_name"] == params["task_name"]

    def test_measure_power_secs(self, online_carbon_tracker, offline_carbon_tracker):
        assert online_carbon_tracker._conf["measure_power_secs"] == 1
        assert offline_carbon_tracker._conf["measure_power_secs"] == 1
