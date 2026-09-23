"""Unit tests for the shared solaros pipeline code on small synthetic frames."""

import numpy as np
import pandas as pd
import pytest

from solaros.constants import CAPACITY_KW, make_location
from solaros.features import build_residual_dataset
from solaros.metrics import interval_coverage, mae, pinball_loss, rmse
from solaros.models import QUANTILES, predict_quantile_generation, train_quantile_models
from solaros.splits import chronological_split


@pytest.fixture(scope="module")
def unified():
    """A full synthetic 2018 hourly frame with the columns the notebooks produce."""
    idx = pd.date_range("2018-01-01", "2018-12-31 23:00", freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    clearsky = make_location().get_clearsky(idx)["ghi"]
    physics = (clearsky / 1000 * 5).clip(upper=CAPACITY_KW)
    actual = (physics * rng.uniform(0.6, 1.1, len(idx))).where(rng.random(len(idx)) > 0.02)
    return pd.DataFrame({
        "temperature_2m": rng.normal(20, 8, len(idx)),
        "shortwave_radiation": clearsky * rng.uniform(0.3, 1.0, len(idx)),
        "physics_predicted_pv": physics,
        "actual_pv_yield_kwh": actual,
    }, index=idx)


def test_features_keep_daytime_rows_with_a_known_residual(unified):
    """X/y hold only daytime hours with a non-NaN residual, and no target columns leak in."""
    features, X, y = build_residual_dataset(unified, make_location())
    assert len(features) == len(unified)
    assert (unified.loc[X.index, "physics_predicted_pv"] > 0).all()
    assert y.notna().all() and X.notna().all().all()
    assert not {"actual_pv_yield_kwh", "household_load_kwh"} & set(X.columns)
    expected = unified.loc[y.index, "actual_pv_yield_kwh"] - unified.loc[y.index, "physics_predicted_pv"]
    pd.testing.assert_series_equal(y, expected)


def test_split_boundaries_are_chronological(unified):
    """Train ends before August, validation covers Aug-Sep, test starts in October."""
    _, X, y = build_residual_dataset(unified, make_location())
    X_train, _, X_val, _, X_test, _ = chronological_split(X, y)
    assert X_train.index.max() < pd.Timestamp("2018-08-01", tz="UTC")
    assert X_val.index.min() >= pd.Timestamp("2018-08-01", tz="UTC")
    assert X_val.index.max() < pd.Timestamp("2018-10-01", tz="UTC")
    assert X_test.index.min() >= pd.Timestamp("2018-10-01", tz="UTC")
    assert len(X_train) + len(X_val) + len(X_test) == len(X)


def test_quantile_generation_is_clipped_and_named(unified):
    """Generation quantiles use q05..q95 column names and stay within [0, capacity]."""
    _, X, y = build_residual_dataset(unified, make_location())
    X_train, y_train, X_val, y_val, X_test, _ = chronological_split(X, y)
    boosters = train_quantile_models(X_train, y_train, X_val, y_val, QUANTILES)
    phys = unified.loc[X_test.index, "physics_predicted_pv"]
    q = predict_quantile_generation(boosters, X_test, phys, CAPACITY_KW)
    assert list(q.columns) == ["q05", "q25", "q50", "q75", "q95"]
    assert (q >= 0).all().all() and (q <= CAPACITY_KW).all().all()


def test_metrics_on_known_values():
    """Hand-checked values for each metric."""
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    b = pd.Series([1.0, 2.0, 3.0, 6.0])
    assert rmse(a, b) == pytest.approx(1.0)
    assert mae(a, b) == pytest.approx(0.5)
    assert interval_coverage(a, a - 0.5, a + 0.5) == 1.0
    assert interval_coverage(a, a + 1, a + 2) == 0.0
    # Under-prediction by 1 at q=0.9 costs 0.9; over-prediction by 1 costs 0.1.
    assert pinball_loss(pd.Series([1.0]), pd.Series([0.0]), 0.9) == pytest.approx(0.9)
    assert pinball_loss(pd.Series([0.0]), pd.Series([1.0]), 0.9) == pytest.approx(0.1)
