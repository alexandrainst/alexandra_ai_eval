"""Unit tests for the `enums` module."""

import enum

from alexandra_ai_eval.country_codes import ALL_COUNTRY_CODES
from alexandra_ai_eval.enums import CountryCode, Device, Framework


class TestDevice:
    def test_is_enum(self):
        assert isinstance(Device, enum.EnumMeta)

    def test_device_list(self):
        assert list(Device.__members__.keys()) == ["CPU", "MPS", "CUDA"]


class TestFramework:
    def test_is_enum(self):
        assert isinstance(Framework, enum.EnumMeta)

    def test_framework_list(self):
        assert list(Framework.__members__.keys()) == ["PYTORCH", "JAX", "SPACY"]


class TestCountryCode:
    def test_is_enum(self):
        assert isinstance(CountryCode, enum.EnumMeta)

    def test_country_code_enum_keys_are_country_codes(self):
        assert list(CountryCode.__members__.keys()) == ALL_COUNTRY_CODES + ["EMPTY"]

    def test_country_code_enum_values_are_country_codes(self):
        enum_values = [obj.value for obj in CountryCode.__members__.values()]
        assert enum_values == ALL_COUNTRY_CODES + [""]
