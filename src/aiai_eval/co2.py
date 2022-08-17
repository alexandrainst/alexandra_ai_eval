"""Functions related to carbon emission measurement."""

from typing import Union

from codecarbon import EmissionsTracker, OfflineEmissionsTracker

from .exceptions import MissingCountryISOCode
from .utils import internet_connection_available


def get_carbon_tracker(
    task_name: str, country_iso_code: str, measure_power_secs: int, verbose: bool
) -> Union[EmissionsTracker, OfflineEmissionsTracker]:
    """Prepares a carbon emissions tracker.

    Args:
        task_name (str):
            Name of the task.
        country_iso_code (str):
            ISO code of the country. Only relevant if no internet connection is
            available. A list of all such codes are available here:
            https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
        measure_power_secs (int):
            How often power is measured, in seconds.
        verbose (bool):
            Whether to print verbose output.

    Returns:
        EmissionsTracker or OfflineEmissionsTracker:
            A carbon emissions tracker. OfflineEmissionsTracker is returned if no
            internet connection is available.
    """
    log_level = "info" if verbose else "error"

    if internet_connection_available():
        carbon_tracker = EmissionsTracker(
            project_name=task_name,
            measure_power_secs=measure_power_secs,
            log_level=log_level,
            save_to_file=False,
            save_to_api=False,
            save_to_logger=False,
        )
    else:
        # If country_iso_code is "", raise exception
        if country_iso_code == "":
            raise MissingCountryISOCode

        carbon_tracker = OfflineEmissionsTracker(
            project_name=task_name,
            measure_power_secs=measure_power_secs,
            country_iso_code=country_iso_code,
            log_level=log_level,
            save_to_file=False,
            save_to_api=False,
            save_to_logger=False,
        )
    return carbon_tracker
