"""Functions related to carbon emission measurement."""

from codecarbon import EmissionsTracker, OfflineEmissionsTracker

from .enums import CountryCode
from .exceptions import MissingCountryISOCode
from .utils import internet_connection_available


def get_carbon_tracker(
    task_name: str,
    country_code: CountryCode,
    verbose: bool,
    prefer_offline: bool = False,
) -> EmissionsTracker | OfflineEmissionsTracker:
    """Prepares a carbon emissions tracker.

    Args:
        task_name:
            Name of the task.
        country_code:
            ISO code of the country. Only relevant if no internet connection is
            available. A list of all such codes are available here:
            https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
        verbose:
            Whether to print verbose output.
        prefer_offline:
            Whether to prefer offline carbon emissions tracker. Defaults to False.

    Returns:
        A carbon emissions tracker. OfflineEmissionsTracker is returned if no internet
        connection is available.

    Raises:
        MissingCountryISOCode:
            If no country code is provided and no internet connection is available.
    """
    log_level = "info" if verbose else "error"

    # Use the offline emissions tracker if there is either no internet connection
    # or the user wants to use the offline emissions tracker
    if not internet_connection_available() or prefer_offline:
        # If the country code is not specified then raise an error
        if country_code == CountryCode.EMPTY:  # type: ignore[attr-defined]
            raise MissingCountryISOCode

        carbon_tracker = OfflineEmissionsTracker(
            project_name=task_name,
            measure_power_secs=1,
            country_iso_code=country_code.value,
            log_level=log_level,
            save_to_file=False,
            save_to_api=False,
            save_to_logger=False,
        )

    # Otherwise use the online emissions tracker
    else:
        carbon_tracker = EmissionsTracker(
            project_name=task_name,
            measure_power_secs=1,
            log_level=log_level,
            save_to_file=False,
            save_to_api=False,
            save_to_logger=False,
        )

    return carbon_tracker
