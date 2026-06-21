# Ensemble Learning — Predicting Demand for Shared Urban Transport

This is a lightweight demo web application that demonstrates using ensemble learning to predict demand in shared urban transport systems (bikes/scooters/etc.). The app is built with Streamlit and includes a training module that generates synthetic data, trains an ensemble of regressors, and saves the model.

Features
- Generate synthetic demand data
- Train an ensemble model (RandomForest + GradientBoosting combined)
- Save/load model
- Simple UI to make single-row predictions

Quick start (Windows PowerShell)

1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, either run PowerShell as Administrator and run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

Or avoid activation and install/run using the venv python directly (shown below).

2. Install dependencies (no activation required):

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

3. Run the Streamlit app (no activation required):

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

What to expect
- Use the sidebar controls to generate sample data, train a model, load an existing model, and run predictions.

Next steps
- Replace synthetic data with your real dataset and adjust feature engineering
- Add cross-validation and hyperparameter tuning
- Add a simple test suite and CI pipeline

License: MIT (feel free to adapt)
