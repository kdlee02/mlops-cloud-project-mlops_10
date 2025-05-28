import pandas as pd
import joblib
from datetime import timedelta
from tqdm import tqdm
from icecream import ic
from src.utils.utils import dataset_dir

def predict_future(model_path, last_date, days=7, save_name='future_temperature.csv'):
    model = joblib.load(model_path)
    future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=days * 24, freq='H')
    future_df = pd.DataFrame({'ds': future_dates})
    ic(f"Predicting {len(future_dates)} future time points")
    forecast = model.predict(future_df)
    result = pd.DataFrame({
        'datetime': future_dates,
        'pred_temp': forecast['yhat'],
        'temp_min': forecast['yhat_lower'],
        'temp_max': forecast['yhat_upper'],
    })
    save_path = f"{dataset_dir()}/{save_name}"
    result.to_csv(save_path, index=False)
    ic(f"Saved future predictions to {save_path}")
    return save_path
