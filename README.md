# UNBURNT prototype

Uncertainty from wildfire hazard to built-environment response, adaptation and insurability.

- Hazard: bootstrap random-forest ensemble, isotonic calibration, split conformal prediction.
- Response/Adaptation/Insurability: illustrative Monte Carlo propagation (placeholder parameters, not calibrated).
- Data: synthetic unless `data.csv` (temperature, wind, humidity, drought, fire) is added.

Run locally: `pip install -r requirements.txt && streamlit run streamlit_app.py`
