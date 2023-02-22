"""Unit tests for the `country_codes` module."""

from alexandra_ai_eval.country_codes import ALL_COUNTRY_CODES


def test_country_codes_list_is_a_list():
    assert isinstance(ALL_COUNTRY_CODES, list)


def test_country_codes_list_contains_three_letter_strings():
    for country_code in ALL_COUNTRY_CODES:
        assert isinstance(country_code, str)
        assert len(country_code) == 3


def test_country_codes_are_all_upper_case():
    for country_code in ALL_COUNTRY_CODES:
        assert country_code.isupper()
