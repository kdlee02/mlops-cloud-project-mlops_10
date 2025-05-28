import pandas as pd
from tqdm import tqdm
from icecream import ic
from src.utils.utils import dataset_dir

def load_and_split(csv_name='tokyo_weather.csv', test_size=0.2):
    df = pd.read_csv(f"{dataset_dir()}/{csv_name}", parse_dates=['time'])
    df = df.sort_values('time')
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    train_path = f"{dataset_dir()}/train.csv"
    test_path = f"{dataset_dir()}/test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    ic(f"Train rows: {len(train)}, Test rows: {len(test)}")
    return train_path, test_path
