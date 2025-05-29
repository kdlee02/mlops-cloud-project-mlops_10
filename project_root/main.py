import fire
import pandas as pd
from icecream import ic

from src.data_loader.data_loader import collect_tokyo_weather
from src.preprocess.preprocess import load_and_split
from src.train.train import train_prophet
from src.evaluate.evaluate import evaluate_prophet
from src.test.test import predict_future
from src.recommend.recommend import recommend_clothing
from src.utils.utils import init_seed
from src.config_loader import load_config

# 1. 전체 파이프라인 실행 (config 사용)
def run_all(config_path='config.yaml'):
    config = load_config(config_path)
    pipeline_cfg = config.get('pipeline', {})
    prophet_cfg = config.get('prophet', {})
    years = pipeline_cfg.get('years', 3)
    days = pipeline_cfg.get('days', 7)
    seed = pipeline_cfg.get('seed', 0)

    init_seed(seed)
    ic("Pipeline started")

    # 1. 데이터 수집
    csv_path = collect_tokyo_weather(years=years)

    # 2. 데이터 분할
    train_csv, test_csv = load_and_split()

    # 3. 모델 학습 (Prophet 파라미터 전달)
    model_path = train_prophet(train_csv, **prophet_cfg)

    # 4. 모델 평가
    metrics = evaluate_prophet(model_path, test_csv)
    print(f"Model MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")

    # 5. 미래 예측
    last_date = pd.read_csv(test_csv, parse_dates=['time'])['time'].max()
    future_csv = predict_future(model_path, last_date, days)

    # 6. 옷차림 추천
    rec_path = recommend_clothing(future_csv.split('/')[-1])
    ic(f"Pipeline complete, recommendations at {rec_path}")

# 2. 각 기능별 Fire 엔트리포인트 제공 (디버깅/단계별 실행)
def collect_weather_cli(years=3):
    return collect_tokyo_weather(years=years)

def split_data_cli(csv_name='tokyo_weather.csv', test_size=0.2):
    return load_and_split(csv_name, test_size)

def train_prophet_cli(train_csv, **kwargs):
    return train_prophet(train_csv, **kwargs)

def evaluate_prophet_cli(model_path, test_csv):
    return evaluate_prophet(model_path, test_csv)

def predict_future_cli(model_path, last_date, days=7):
    return predict_future(model_path, pd.to_datetime(last_date), days)

def recommend_clothing_cli(future_csv='future_temperature.csv'):
    return recommend_clothing(future_csv)

if __name__ == '__main__':
    fire.Fire({
        'run_all': run_all,
        'collect_weather': collect_weather_cli,
        'split_data': split_data_cli,
        'train_prophet': train_prophet_cli,
        'evaluate_prophet': evaluate_prophet_cli,
        'predict_future': predict_future_cli,
        'recommend_clothing': recommend_clothing_cli,
    })
