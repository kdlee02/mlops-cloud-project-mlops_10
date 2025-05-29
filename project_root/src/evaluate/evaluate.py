import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from icecream import ic

def evaluate_prophet(model_path, test_csv):
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv, parse_dates=['time'])
    test_data = df[['time', 'temp']].copy()
    test_data.columns = ['ds', 'y']
    forecast = model.predict(test_data[['ds']])
    mae = mean_absolute_error(test_data['y'], forecast['yhat'])
    rmse = mean_squared_error(test_data['y'], forecast['yhat'], squared=False)
    ic(mae, rmse)
    return {'mae': mae, 'rmse': rmse}
