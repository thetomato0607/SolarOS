# SolarOS: A Hardware-Agnostic AI Energy Optimisation Platform

SolarOS is an autonomous home energy intelligence platform designed to maximise the solar self-consumption and minimise household utility costs under dynamic electricity tariffs.

Unlike brand-specific, walled-garden energy applications, SolarOS serves as a hardware-agnostic orchestration layer sitting above dispararte asset classes (inverter, home batteries, EV charges and smart loads). It couples a Physics-Guided Gray-Box Machine Learning forecasting engine with a Mixed-Integer Linear Programming (MILP) optimisation solver to autonomously schedule residential energy flows.

## Core Technical Innovations

### 1. Physics-Guided Residual Forecasting 

Pure mahcine learning models often struggle with bounded physical systems, producing physically impossible predictions (e.g. solar generation at midnight). Pure physics models fail to capture localised realities like micro-shading. SolarOS resolves this using a Gray-Box Residual Framework:

- Physical Baseline: Uses pvlib-python to compute theoretical clear-sky and transposition irradiance based on localised solar geometry (azimuth, tilt and elevation) and panel characteristics

- Machine Learning Error Correction: A LightGBM regressor accpets historical weather transients (cloud cover, temperature) to predict the error delta (residual) of the physics models.

- Probabilistic Uncertainty Estimation: The machine lerning engine uses Quantile Loss ($a$ = [0.1, 0.5, 0.9]) to yield risk-aware predictive distributions ($P_10, P_50, P_90$)