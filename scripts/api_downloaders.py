"""
Contains one downloader function per external data API used in this
project. Each function is responsible for exactly one API: its base URL
and any fixed/static query parameters are hardcoded inside the function itself.

Currently implemented:
    - download_open_meteo_historical(): Open-Meteo Historical Weather API

Adding a new data source means adding one new function here, 
following the same pattern.
"""

# Required packages
from __future__ import annotations
import logging
from datetime import date
import requests

logger = logging.getLogger(__name__)

class DownloaderError(Exception):
    """Raised when an API request fails."""


# ---------------------------------------------------------------------------
# Open-Meteo Historical Weather API
# ---------------------------------------------------------------------------
def download_open_meteo_historical(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timeout: int = 30,
) -> dict:
    """
    Downloads hourly historical weather data for a single location and
    date range from the Open-Meteo Historical Weather API.
    The API endpoint, the list of requested hourly variables, and the
    unit system are hardcoded here. Only the location and date range 
    are function parameters.

    Returns:
        The parsed JSON response as a dictionary. The hourly values are
        found under response["hourly"], as a dict of parallel lists
        (one list per variable, all aligned with response["hourly"]["time"]).

    Raises:
        DownloaderError: If the request fails, times out, or the API
            returns a non-successful HTTP status code.
    """
    
    # API configuration
    url = "https://archive-api.open-meteo.com/v1/archive"
    hourly_variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "pressure_msl",
        "cloud_cover",
        "windspeed_10m",
        "windspeed_80m",
        "windspeed_120m",
        "windspeed_180m",
        "winddirection_10m",
        "shortwave_radiation",
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(hourly_variables),
        "timezone": "UTC",
        "windspeed_unit": "kmh",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # Wrap any network-level or HTTP-status error
        raise DownloaderError(
            f"Open-Meteo request failed for lat={latitude}, lon={longitude}, "
            f"{start_date} to {end_date}: {e}"
        ) from e

    return response.json()