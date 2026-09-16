"""
test_downloaders.py

Tests for api_downloaders.py and historic_data_downloader.py.

All tests mock the network layer (requests.get / download_open_meteo_historical)
so they run instantly, work offline, and never hit real API rate limits.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
import requests

from api_downloaders import DownloaderError, download_open_meteo_historical
from historic_data_downloader import (
    LOCATIONS,
    _validate_response,
    download_location_range,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hourly_payload(num_hours: int) -> dict:
    """Builds a minimal, well-formed Open-Meteo-style response body."""
    return {
        "hourly": {
            "time": [f"2024-01-01T{h:02d}:00" for h in range(num_hours)],
            "temperature_2m": [10.0] * num_hours,
            "relative_humidity_2m": [80.0] * num_hours,
            "precipitation": [0.0] * num_hours,
            "rain": [0.0] * num_hours,
            "snowfall": [0.0] * num_hours,
            "pressure_msl": [1013.0] * num_hours,
            "cloud_cover": [50.0] * num_hours,
            "windspeed_10m": [15.0] * num_hours,
            "windspeed_80m": [20.0] * num_hours,
            "windspeed_120m": [22.0] * num_hours,
            "windspeed_180m": [24.0] * num_hours,
            "winddirection_10m": [270.0] * num_hours,
            "shortwave_radiation": [100.0] * num_hours,
        }
    }


# ---------------------------------------------------------------------------
# api_downloaders.download_open_meteo_historical
# ---------------------------------------------------------------------------

class TestDownloadOpenMeteoHistorical:

    @patch("api_downloaders.requests.get")
    def test_returns_parsed_json_on_success(self, mock_get):
        payload = _make_hourly_payload(num_hours=24)
        mock_response = MagicMock()
        mock_response.json.return_value = payload
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = download_open_meteo_historical(
            latitude=52.52,
            longitude=13.41,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert result == payload

    @patch("api_downloaders.requests.get")
    def test_sends_expected_url_and_params(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = _make_hourly_payload(24)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        download_open_meteo_historical(
            latitude=52.52,
            longitude=13.41,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
        )

        called_url = mock_get.call_args.args[0]
        called_params = mock_get.call_args.kwargs["params"]

        assert called_url == "https://archive-api.open-meteo.com/v1/archive"
        assert called_params["latitude"] == 52.52
        assert called_params["longitude"] == 13.41
        assert called_params["start_date"] == "2024-01-01"
        assert called_params["end_date"] == "2024-01-02"
        assert "temperature_2m" in called_params["hourly"]

    @patch("api_downloaders.requests.get")
    def test_raises_downloader_error_on_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        mock_get.return_value = mock_response

        with pytest.raises(DownloaderError):
            download_open_meteo_historical(
                latitude=52.52,
                longitude=13.41,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1),
            )

    @patch("api_downloaders.requests.get")
    def test_raises_downloader_error_on_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("no network")

        with pytest.raises(DownloaderError):
            download_open_meteo_historical(
                latitude=52.52,
                longitude=13.41,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1),
            )


# ---------------------------------------------------------------------------
# historic_data_downloader._validate_response
# ---------------------------------------------------------------------------

class TestValidateResponse:

    def test_passes_for_well_formed_single_day_response(self):
        data = _make_hourly_payload(num_hours=24)
        # Should not raise.
        _validate_response(data, date(2024, 1, 1), date(2024, 1, 1))

    def test_passes_for_well_formed_multi_day_response(self):
        data = _make_hourly_payload(num_hours=48)
        # Should not raise.
        _validate_response(data, date(2024, 1, 1), date(2024, 1, 2))

    def test_raises_when_hourly_key_missing(self):
        with pytest.raises(AssertionError):
            _validate_response({}, date(2024, 1, 1), date(2024, 1, 1))

    def test_raises_when_hour_count_does_not_match_date_range(self):
        # Only 20 hours instead of the expected 24 for a single day.
        data = _make_hourly_payload(num_hours=20)
        with pytest.raises(AssertionError):
            _validate_response(data, date(2024, 1, 1), date(2024, 1, 1))

    def test_raises_when_a_variable_array_length_mismatches(self):
        data = _make_hourly_payload(num_hours=24)
        # Truncate one variable so it no longer aligns with 'time'.
        data["hourly"]["rain"] = data["hourly"]["rain"][:-5]
        with pytest.raises(AssertionError):
            _validate_response(data, date(2024, 1, 1), date(2024, 1, 1))


# ---------------------------------------------------------------------------
# historic_data_downloader.download_location_range
# ---------------------------------------------------------------------------

class TestDownloadLocationRange:

    @patch("historic_data_downloader.download_open_meteo_historical")
    def test_returns_dataframe_with_location_columns(self, mock_download):
        mock_download.return_value = _make_hourly_payload(num_hours=24)

        df = download_location_range(
            location_name="berlin_de",
            latitude=52.52,
            longitude=13.41,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert len(df) == 24
        assert (df["location_name"] == "berlin_de").all()
        assert (df["latitude"] == 52.52).all()
        assert (df["longitude"] == 13.41).all()
        assert "temperature_2m" in df.columns

    @patch("historic_data_downloader.download_open_meteo_historical")
    def test_raises_if_validation_fails(self, mock_download):
        # Wrong hour count for the requested single-day range.
        mock_download.return_value = _make_hourly_payload(num_hours=10)

        with pytest.raises(AssertionError):
            download_location_range(
                location_name="berlin_de",
                latitude=52.52,
                longitude=13.41,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1),
            )


# ---------------------------------------------------------------------------
# LOCATIONS sanity check
# ---------------------------------------------------------------------------

def test_locations_cover_expected_regions():
    # A lightweight guard against accidentally deleting entries: makes
    # sure the location set still includes representatives of every
    # region the project is meant to cover.
    expected_locations = {
        "berlin_de", "paris_fr", "warsaw_pl", "amsterdam_nl", "oslo_no",
        "reykjavik_is", "london_gb", "nuuk_gl", "moscow_ru", "toronto_ca",
        "cairo_eg", "casablanca_ma",
    }
    assert expected_locations.issubset(LOCATIONS.keys())