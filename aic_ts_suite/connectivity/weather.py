"""
Weather data integration via the Open-Meteo API.

Inspired by https://github.com/AntoinePinto/weather-data

Provides:
    fetch_weather()          – fetch historical or forecast weather for a location
    merge_weather()          – join weather columns into an existing DataFrame
    AVAILABLE_VARIABLES      – set of variable names accepted by Open-Meteo

Open-Meteo is free to use, requires no API key, and covers worldwide locations.

Archive endpoint  : https://archive-api.open-meteo.com/v1/archive
Forecast endpoint : https://api.open-meteo.com/v1/forecast

Usage (standalone)
------------------
>>> from aic_ts_suite.connectivity.weather import fetch_weather, merge_weather
>>> weather_df = fetch_weather(
...     latitude=47.8095, longitude=13.0550,
...     start_date="2020-01-01", end_date="2023-12-31",
...     variables=["temperature_2m", "precipitation"],
... )
>>> merged = merge_weather(main_df, weather_df, freq="MS", how="left")
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AVAILABLE_VARIABLES: set[str] = {
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
}

# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_weather(
    latitude: float,
    longitude: float,
    variables: List[str],
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None,
    source: str = "archive",
    forecast_days: int = 16,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo for a single location.

    Parameters
    ----------
    latitude, longitude : float
        WGS84 coordinates of the target location.
    variables : list[str]
        Open-Meteo hourly variable names (see AVAILABLE_VARIABLES).
    start_date, end_date : str | date, optional
        Required for ``source="archive"``.  Format: "YYYY-MM-DD".
    source : {"archive", "forecast", "auto"}
        "archive"  – historical data (requires start_date / end_date).
        "forecast" – next N days.
        "auto"     – archive for past dates, forecast for future dates.
    forecast_days : int
        Number of forecast days when source="forecast" (max 16).
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (hourly, UTC), one column per requested variable.
        Returns an empty DataFrame on failure.
    """
    unknown = set(variables) - AVAILABLE_VARIABLES
    if unknown:
        logger.warning("Unknown weather variables will be ignored: %s", unknown)
        variables = [v for v in variables if v in AVAILABLE_VARIABLES]
    if not variables:
        raise ValueError("No valid weather variables specified.")

    if source == "auto":
        today = date.today()
        sd = pd.to_datetime(start_date).date() if start_date else today - timedelta(days=365)
        ed = pd.to_datetime(end_date).date() if end_date else today + timedelta(days=forecast_days)
        archive_end = min(ed, today - timedelta(days=5))
        frames = []
        if sd <= archive_end:
            frames.append(
                fetch_weather(latitude, longitude, variables,
                              start_date=sd, end_date=archive_end,
                              source="archive", timeout=timeout)
            )
        if ed >= today:
            frames.append(
                fetch_weather(latitude, longitude, variables,
                              source="forecast", forecast_days=forecast_days, timeout=timeout)
            )
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined

    if source == "forecast":
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(variables),
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        url = FORECAST_URL
    else:  # archive
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required for source='archive'.")
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(variables),
            "start_date": str(start_date)[:10],
            "end_date": str(end_date)[:10],
            "timezone": "UTC",
        }
        url = ARCHIVE_URL

    logger.info(
        "Fetching weather [%s] lat=%.4f lon=%.4f vars=%s",
        source, latitude, longitude, variables,
    )

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("Weather API request failed: %s", exc)
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    if "time" not in hourly:
        logger.error("Unexpected API response structure: %s", list(data.keys()))
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").rename_axis("timestamp")

    logger.info("Weather fetched: %d rows × %d vars", len(df), len(df.columns))
    return df


# ── Aggregation / resampling ──────────────────────────────────────────────────

_DEFAULT_AGG: dict[str, str] = {
    "temperature_2m": "mean",
    "apparent_temperature": "mean",
    "dew_point_2m": "mean",
    "relative_humidity_2m": "mean",
    "precipitation": "sum",
    "rain": "sum",
    "snowfall": "sum",
    "snow_depth": "mean",
    "wind_speed_10m": "mean",
    "wind_gusts_10m": "max",
    "wind_direction_10m": "mean",
    "cloud_cover": "mean",
    "pressure_msl": "mean",
    "surface_pressure": "mean",
    "shortwave_radiation": "sum",
    "direct_radiation": "sum",
    "diffuse_radiation": "sum",
    "et0_fao_evapotranspiration": "sum",
    "vapour_pressure_deficit": "mean",
    "soil_temperature_0_to_7cm": "mean",
    "soil_moisture_0_to_7cm": "mean",
    "weather_code": "max",
}


def resample_weather(
    weather_df: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """
    Resample hourly weather data to *freq* using sensible aggregations.

    For downsampling (e.g. hourly → daily/monthly) each variable is
    aggregated with the most appropriate function (sum for precipitation,
    mean for temperature, etc.).

    For upsampling (e.g. hourly → 15min) the data is reindexed to the
    target frequency and forward-filled.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Output of ``fetch_weather()`` (hourly, UTC DatetimeIndex).
    freq : str
        Pandas offset alias, e.g. "15min", "30min", "H", "D", "MS".

    Returns
    -------
    pd.DataFrame
        Resampled weather features.
    """
    import pandas as pd

    # Determine whether this is an up- or down-sample relative to hourly
    try:
        target_delta = pd.tseries.frequencies.to_offset(freq)
        hourly_delta = pd.tseries.frequencies.to_offset("H")
        is_upsample = target_delta < hourly_delta  # type: ignore[operator]
    except Exception:
        is_upsample = False

    if is_upsample:
        # Upsample: reindex to finer grid and forward-fill
        new_idx = pd.date_range(
            start=weather_df.index.min(),
            end=weather_df.index.max(),
            freq=freq,
            tz=weather_df.index.tz,
        )
        resampled = weather_df.reindex(new_idx).ffill()
    else:
        agg = {
            col: _DEFAULT_AGG.get(col, "mean")
            for col in weather_df.columns
        }
        resampled = weather_df.resample(freq).agg(agg)

    resampled.index = resampled.index.tz_localize(None)  # strip UTC for merging
    return resampled


# ── Merge helper ──────────────────────────────────────────────────────────────

def _to_utc_naive(ts) -> pd.Series:
    """Convert a datetime Series or Index to UTC-naive (strips tz if present)."""
    s = pd.to_datetime(ts)
    if hasattr(s, "dt"):
        if s.dt.tz is not None:
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s
    else:  # Index
        if s.tz is not None:
            return s.tz_convert("UTC").tz_localize(None)
        return s


def merge_weather(
    main_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    freq: str,
    how: str = "left",
    prefix: str = "weather_",
) -> pd.DataFrame:
    """
    Join weather features into *main_df* using nearest-prior-match alignment.

    For frequencies finer than hourly (e.g. 15-min) ``pd.merge_asof`` is used
    so each data row receives the weather observation from the most recent
    completed hour — no upsampling required and no NaN gaps.

    For coarser frequencies (daily, monthly, etc.) the weather is first
    downsampled with ``resample_weather`` before joining on index.

    Parameters
    ----------
    main_df : pd.DataFrame
        Main dataset.  Must have a ``timestamp`` column or a DatetimeIndex.
    weather_df : pd.DataFrame
        Hourly weather DataFrame from ``fetch_weather()``.
    freq : str
        Target frequency for resampling when downsampling is needed.
    how : str
        Join type for coarse-frequency index joins (default "left").
    prefix : str
        Column prefix added to all weather columns (default: "weather_").

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with weather columns appended.
    """
    # Determine whether data frequency is finer than hourly
    # Try both old ("H") and new ("h") pandas aliases for robustness
    try:
        target_delta = pd.tseries.frequencies.to_offset(freq)
        try:
            hourly_delta = pd.tseries.frequencies.to_offset("h")
        except Exception:
            hourly_delta = pd.tseries.frequencies.to_offset("H")
        is_sub_hourly = target_delta < hourly_delta  # type: ignore[operator]
    except Exception:
        is_sub_hourly = False

    has_ts_col = "timestamp" in main_df.columns

    # Build UTC-naive weather table
    w = weather_df.copy()
    w.index = _to_utc_naive(w.index)
    w_reset = w.reset_index()                        # → "timestamp" column
    w_reset.columns = ["_w_ts"] + [f"{prefix}{c}" for c in w.columns]
    w_reset = w_reset.sort_values("_w_ts").reset_index(drop=True)
    prefixed_cols = [c for c in w_reset.columns if c.startswith(prefix)]

    # Build UTC-naive main timestamps
    if has_ts_col:
        main_ts = _to_utc_naive(main_df["timestamp"])
    else:
        main_ts = _to_utc_naive(pd.Series(main_df.index))

    if is_sub_hourly:
        # merge_asof: each 15-min row inherits the weather from the last hour
        left = main_df.copy().reset_index(drop=True)
        left["_m_ts"] = main_ts.values
        left_sorted = left.sort_values("_m_ts").reset_index(drop=True)
        orig_order = left.sort_values("_m_ts").index  # to restore original row order

        merged_sorted = pd.merge_asof(
            left_sorted,
            w_reset,
            left_on="_m_ts",
            right_on="_w_ts",
            direction="backward",
        ).drop(columns=["_m_ts", "_w_ts"])

        # Restore original row order
        merged = merged_sorted.loc[orig_order.argsort()].reset_index(drop=True)

    else:
        # Downsample then index-join
        resampled = resample_weather(weather_df, freq)
        resampled.columns = [f"{prefix}{c}" for c in resampled.columns]
        resampled.index = _to_utc_naive(resampled.index)

        main_indexed = main_df.copy()
        if has_ts_col:
            main_indexed = main_indexed.drop(columns=["timestamp"])
        main_indexed.index = main_ts.values
        merged = main_indexed.join(resampled, how=how).reset_index(drop=True)

        if has_ts_col:
            merged.insert(0, "timestamp", main_df["timestamp"].values)

    n_matched = merged[prefixed_cols].notna().any(axis=1).sum()
    logger.info(
        "Weather merged: %d/%d rows matched (%d new cols)",
        n_matched, len(merged), len(prefixed_cols),
    )
    return merged
