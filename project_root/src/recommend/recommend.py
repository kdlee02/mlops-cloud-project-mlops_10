import pandas as pd
from tqdm import tqdm
from icecream import ic
from src.utils.utils import dataset_dir

def recommend_clothing(future_csv='future_temperature.csv', save_name='clothing_recommendation.csv'):
    df = pd.read_csv(f"{dataset_dir()}/{future_csv}", parse_dates=['datetime'])
    df['date'] = df['datetime'].dt.date
    daily = df.groupby('date').agg({
        'pred_temp': ['min', 'max', 'mean']
    })
    daily.columns = ['min_temp', 'max_temp', 'avg_temp']
    daily = daily.reset_index()

    def recommend(temp):
        if temp >= 28:
            return '반팔, 반바지, 샌들 (매우 더움)'
        elif temp >= 23:
            return '반팔, 긴바지, 운동화 (더움)'
        elif temp >= 18:
            return '긴팔, 긴바지 (적당함)'
        elif temp >= 12:
            return '긴팔, 니트, 자켓 (쌀쌀함)'
        elif temp >= 5:
            return '코트, 니트, 긴바지 (추움)'
        else:
            return '두꺼운 코트, 목도리 (매우 추움)'

    tqdm.pandas(desc="옷차림 추천 생성")
    daily['clothing'] = daily['avg_temp'].progress_apply(recommend)
    save_path = f"{dataset_dir()}/{save_name}"
    daily[['date', 'min_temp', 'max_temp', 'avg_temp', 'clothing']].to_csv(save_path, index=False)
    ic(f"Saved clothing recommendations to {save_path}")
    return save_path
