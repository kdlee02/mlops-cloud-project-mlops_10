import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from icecream import ic

def evaluate_prophet(model_path, test_csv, run_id):
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv, parse_dates=['time'])
    test_data = df[['time', 'temp']].copy()
    test_data.columns = ['ds', 'y']
    forecast = model.predict(test_data[['ds']])
    mae = mean_absolute_error(test_data['y'], forecast['yhat'])
    rmse = mean_squared_error(test_data['y'], forecast['yhat'], squared=False)
    ic(mae, rmse)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
    return {'mae': mae, 'rmse': rmse}

def evaluate_sarimax(model_path, test_csv, run_id):
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv, parse_dates=['time'])
    y_true = df['temp'].values
    start = 0
    end = len(y_true) - 1

    ic("Running SARIMAX forecast...")
    y_pred = model.predict(start=start, end=end)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    ic(mae, rmse)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
    return {'mae': mae, 'rmse': rmse}