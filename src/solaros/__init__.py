"""Shared pipeline code for the SolarOS notebooks.

Modules:
    constants: site coordinates, array geometry and nameplate capacity.
    features: residual-model feature matrix built from the unified parquet.
    splits: chronological train/validation/test split.
    models: LightGBM settings and the quantile-model ensemble.
    metrics: error, coverage and pinball-loss metrics.
"""
