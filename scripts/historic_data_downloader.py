"""
Calls the downloader functions in api_downloaders.py for a given date
range, across a fixed set of locations relevant to short-term weather
forecasting. Includes sanity checks on the returned data, and an option
to save the result directly into the project's SQLite database.

This script is intended to be run manually whenever historic data needs 
to be downloaded.
"""

# Packages
from __future__ import annotations
import logging
import sqlite3
from datetime import date
from pathlib import Path
import pandas as pd
from api_downloaders import DownloaderError, download_open_meteo_historical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Location selection
# ---------------------------------------------------------------------------
# Covers Germany (the forecast target) plus surrounding/contributing
# regions chosen for their influence on Central European weather systems.
# Format: location_name -> (latitude, longitude)
LOCATIONS: dict[str, tuple[float, float]] = {
    "berlin_de": (52.52, 13.41),
    "paris_fr": (48.8566, 2.3522),
    "warsaw_pl": (52.2297, 21.0122),
    "amsterdam_nl": (52.3676, 4.9041),
    "oslo_no": (59.9139, 10.7522),
    "reykjavik_is": (64.1466, -21.9426),
    "london_gb": (51.5074, -0.1278),
    "nuuk_gl": (64.1836, -51.7214),
    "moscow_ru": (55.7558, 37.6173),
    "toronto_ca": (43.6532, -79.3832),
    "cairo_eg": (30.0444, 31.2357),
    "casablanca_ma": (33.5731, -7.5898),
}

# Set database path for storage
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "database" / "weather.db"
TABLE_NAME = "weather_hourly"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_response(data: dict, start_date: date, end_date: date) -> None:
    """
    Runs sanity checks on the raw API response to catch malformed or
    incomplete data early, before it gets turned into a DataFrame or
    written to the database.

    Raises:
        AssertionError: If the response does not match what was expected
            for the requested date range.
    """
    assert "hourly" in data, "Response is missing the 'hourly' section."
    hourly = data["hourly"]

    assert "time" in hourly, "Response is missing 'hourly.time'."

    # Open-Meteo returns one value per hour, covering every hour of every
    # requested day (inclusive on both ends).
    expected_hours = ((end_date - start_date).days + 1) * 24
    actual_hours = len(hourly["time"])
    assert actual_hours == expected_hours, (
        f"Expected {expected_hours} hourly timestamps for "
        f"{start_date} to {end_date}, got {actual_hours}."
    )

    # Every requested variable should be present, and have exactly as
    # many values as there are timestamps - otherwise the columns would
    # not align correctly when building the DataFrame.
    for key, values in hourly.items():
        if key == "time":
            continue
        assert len(values) == actual_hours, (
            f"Variable '{key}' has {len(values)} values, "
            f"expected {actual_hours} (mismatch with 'time')."
        )


# ---------------------------------------------------------------------------
# Downloading a single location
# ---------------------------------------------------------------------------
def download_location_range(
    location_name: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Downloads and validates hourly historical weather data for a single
    location and date range, returning it as a tidy DataFrame.

    Raises:
        DownloaderError: If the underlying API request fails.
        AssertionError: If the response fails the sanity checks.
    """
    logger.info(
        "Downloading %s (%s, %s) from %s to %s",
        location_name, latitude, longitude, start_date, end_date,
    )

    data = download_open_meteo_historical(latitude, longitude, start_date, end_date)
    _validate_response(data, start_date, end_date)

    df = pd.DataFrame(data["hourly"])
    df.insert(0, "longitude", longitude)
    df.insert(0, "latitude", latitude)
    df.insert(0, "location_name", location_name)

    return df


# ---------------------------------------------------------------------------
# Database storage
# ---------------------------------------------------------------------------

def _create_table_if_not_exists(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            location_name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            time TEXT NOT NULL,
            temperature_2m REAL,
            relative_humidity_2m REAL,
            precipitation REAL,
            rain REAL,
            snowfall REAL,
            pressure_msl REAL,
            cloud_cover REAL,
            windspeed_10m REAL,
            windspeed_80m REAL,
            windspeed_120m REAL,
            windspeed_180m REAL,
            winddirection_10m REAL,
            shortwave_radiation REAL,
            PRIMARY KEY (location_name, time)
        );
        """
    )
    conn.commit()


def save_dataframe_to_db(df: pd.DataFrame, db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Saves a location's DataFrame into the SQLite weather_hourly table.

    Uses "INSERT OR REPLACE" so that re-running a download for a date
    range that was already stored updates the existing rows instead of
    creating duplicates - this makes the download safe to re-run.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        _create_table_if_not_exists(conn)

        columns = list(df.columns)
        placeholders = ", ".join("?" for _ in columns)
        column_names = ", ".join(columns)

        conn.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} ({column_names}) "
            f"VALUES ({placeholders});",
            df[columns].itertuples(index=False, name=None),
        )
        conn.commit()
        logger.info("Saved %d rows for %s to %s", len(df), df["location_name"].iloc[0], db_path)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Downloading all locations
# ---------------------------------------------------------------------------

def download_all_locations(
    start_date: date,
    end_date: date,
    save_to_db: bool = False,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, pd.DataFrame]:
    """
    Downloads historic weather data for every location in LOCATIONS over
    the given date range. Locations that fail to download or fail
    validation are logged and skipped, rather than stopping the entire
    batch.

    Args:
        start_date: First date (inclusive) to request data for.
        end_date: Last date (inclusive) to request data for.
        save_to_db: If True, each location's data is written directly to
            the SQLite database as soon as it is downloaded and validated.
        db_path: Path to the SQLite database file (only used if
            save_to_db is True).

    Returns:
        A dictionary mapping location_name -> DataFrame, containing only
        the locations that were downloaded successfully.
    """
    results: dict[str, pd.DataFrame] = {}

    for location_name, (latitude, longitude) in LOCATIONS.items():
        try:
            df = download_location_range(
                location_name, latitude, longitude, start_date, end_date
            )
        except (DownloaderError, AssertionError) as e:
            logger.error("Skipping %s due to error: %s", location_name, e)
            continue

        results[location_name] = df

        if save_to_db:
            save_dataframe_to_db(df, db_path=db_path)

    logger.info(
        "Finished: %d/%d locations downloaded successfully.",
        len(results), len(LOCATIONS),
    )
    return results


# ---------------------------------------------------------------------------
# Manual usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: download one week of historic data for all locations and
    # save it directly to the database. Adjust the dates as needed for
    # a manual run.
    example_start = date(2024, 1, 1)
    example_end = date(2024, 1, 7)

    download_all_locations(example_start, example_end, save_to_db=True)