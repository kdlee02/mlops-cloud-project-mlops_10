import pandas as pd
from meteostat import Point, Hourly
from datetime import datetime, timedelta
from tqdm import tqdm
from icecream import ic
from src.utils.utils import dataset_dir, ensure_dir

def collect_tokyo_weather(years=3, save_name='tokyo_weather.csv'):
    ensure_dir(dataset_dir())
    tokyo = Point(35.6762, 139.6503, 70)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years)
    data = Hourly(tokyo, start_date, end_date).fetch()
    df = data.reset_index()
    df = df.dropna(subset=['temp'])
    save_path = f"{dataset_dir()}/{save_name}"
    df[['time', 'temp']].to_csv(save_path, index=False)
    ic(f"Saved weather data to {save_path}, rows: {len(df)}")
    return save_path
