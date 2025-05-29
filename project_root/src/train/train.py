import pandas as pd
from prophet import Prophet
import joblib
from tqdm import tqdm
from icecream import ic
from src.utils.utils import model_dir, ensure_dir

def train_prophet(
    train_csv,
    model_name='prophet_model.pkl',
    seasonality_mode='multiplicative',
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.01,
    seasonality_prior_scale=15.0,
    **kwargs
):
    ensure_dir(model_dir())
    df = pd.read_csv(train_csv, parse_dates=['time'])
    data = df[['time', 'temp']].copy()
    data.columns = ['ds', 'y']
    model = Prophet(
        seasonality_mode=seasonality_mode,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        **kwargs
    )
    ic("Fitting Prophet model...")
    model.fit(data)
    model_path = model_dir(model_name)
    joblib.dump(model, model_path)
    ic(f"Model saved to {model_path}")
    return model_path
