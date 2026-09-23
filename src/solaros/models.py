"""LightGBM settings and the quantile ensemble used in notebooks 03-05."""

import lightgbm as lgb
import pandas as pd

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# Shared by the point model (03) and every quantile model (04, 05).
LGB_BASE_PARAMS = {
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "seed":              42,
    "verbosity":         -1,
}


def train_quantile_models(X_train, y_train, X_val, y_val, quantiles=QUANTILES,
                          log_period=0, verbose=False) -> dict:
    """Train one LightGBM quantile regressor per level, early-stopped on validation loss.

    Args:
        log_period: Print the validation loss every ``log_period`` rounds (0 = off).
        verbose: Also print a header and the best iteration for each quantile.

    Returns:
        Dict mapping each quantile level to its trained booster.
    """
    boosters = {}
    for q in quantiles:
        params = {
            **LGB_BASE_PARAMS,
            "objective": "quantile",
            "alpha":     q,
            "metric":    "quantile",
        }
        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

        if verbose:
            print(f"--- Training q={q:.2f} ---")
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=log_period),
            ],
        )
        boosters[q] = booster
        if verbose:
            print(f"  Best iter: {booster.best_iteration}  |  "
                  f"Best val quantile loss: {booster.best_score['val']['quantile']:.4f}\n")
    return boosters


def predict_quantile_generation(boosters, X, physics_pv, capacity_kw) -> pd.DataFrame:
    """Convert residual quantile predictions into generation quantiles.

    Adds each predicted residual to the physics forecast and clips to
    [0, capacity_kw]. Columns are named ``q05``, ``q25``, ... in ascending
    order; rows are not re-sorted, so quantiles may still cross.
    """
    cols = {}
    for q, booster in boosters.items():
        resid = booster.predict(X, num_iteration=booster.best_iteration)
        cols[f"q{int(q * 100):02d}"] = (physics_pv + resid).clip(0, capacity_kw)
    return pd.DataFrame(cols, index=X.index)
