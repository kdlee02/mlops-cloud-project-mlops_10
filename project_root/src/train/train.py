import pandas as pd
from prophet import Prophet
import joblib
from tqdm import tqdm
from icecream import ic
from src.utils.utils import model_dir, ensure_dir
#import bentoml

def train_prophet(train_csv, model_name='prophet_model.pkl'):
    ensure_dir(model_dir())
    df = pd.read_csv(train_csv, parse_dates=['time'])
    data = df[['time', 'temp']].copy()
    data.columns = ['ds', 'y']
    model = Prophet(
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=15.0
    )
    ic("Fitting Prophet model...")
    model.fit(data)
    model_path = model_dir(model_name)
    joblib.dump(model, model_path)
    ic(f"Model saved to {model_path}")
    # bento_model = bentoml.models.save_model(name="weather_predictor", 
    # model=model, signatures={"predict": {"batchable": True}})
    return model_path
