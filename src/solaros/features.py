"""Feature matrix for the ML residual model (notebooks 03-05)."""

import numpy as np
import pandas as pd


def build_residual_dataset(unified: pd.DataFrame, location) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build weather, physics and time features plus the residual target.

    Args:
        unified: Hourly frame with ``temperature_2m``, ``shortwave_radiation``,
            ``physics_predicted_pv`` and ``actual_pv_yield_kwh``.
        location: pvlib Location used for solar position and clear-sky GHI.

    Returns:
        ``(features, X, y)``: ``features`` covers every hour; ``X``/``y`` keep
        daytime hours (physics output > 0) whose residual is not NaN. The
        target is ``actual - physics``.
    """
    solar_pos = location.get_solarposition(unified.index)
    clearsky = location.get_clearsky(unified.index)

    # Clearness index GHI / clear-sky GHI, clipped to [0, 1.5]. Hours with
    # clear-sky GHI <= 10 W/m2 (night/twilight) are set to 0 rather than cloud cover.
    kt = (
        unified["shortwave_radiation"]
        / clearsky["ghi"].where(clearsky["ghi"] > 10, other=np.nan)
    ).clip(0, 1.5).fillna(0)

    hour = unified.index.hour
    doy = unified.index.dayofyear

    features = pd.DataFrame({
        # Weather
        "temperature_2m":   unified["temperature_2m"],
        "ghi":              unified["shortwave_radiation"],
        "clearness_index":  kt,
        # Physics
        "physics_pv":       unified["physics_predicted_pv"],
        "clearsky_ghi":     clearsky["ghi"],
        "solar_elevation":  solar_pos["apparent_elevation"],
        # Time, encoded cyclically so 23:00 and 00:00 are neighbours
        "hour_sin":  np.sin(2 * np.pi * hour / 24),
        "hour_cos":  np.cos(2 * np.pi * hour / 24),
        "doy_sin":   np.sin(2 * np.pi * doy / 365),
        "doy_cos":   np.cos(2 * np.pi * doy / 365),
    }, index=unified.index)

    residual = unified["actual_pv_yield_kwh"] - unified["physics_predicted_pv"]

    daytime_mask = unified["physics_predicted_pv"] > 0
    X = features[daytime_mask].copy()
    y = residual[daytime_mask].copy()
    valid_rows = y.notna()
    return features, X[valid_rows], y[valid_rows]
