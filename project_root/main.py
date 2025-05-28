import fire
from src.data_loader.data_loader import collect_tokyo_weather
from src.preprocess.preprocess import load_and_split
from src.train.train import train_prophet
from src.evaluate.evaluate import evaluate_prophet
from src.test.test import predict_future
from src.recommend.recommend import recommend_clothing
from src.utils.utils import init_seed
import pandas as pd
from icecream import ic

def run_all(years=3, days=7, seed=0):
    init_seed(seed)
    ic("Pipeline started")
    # 1. 데이터 수집
    csv_path = collect_tokyo_weather(years=years)
    # 2. 데이터 분할
    train_csv, test_csv = load_and_split()
    # 3. 모델 학습
    model_path = train_prophet(train_csv)
    # 4. 모델 평가
    metrics = evaluate_prophet(model_path, test_csv)
    print(f"Model MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
    # 5. 미래 예측
    last_date = pd.read_csv(test_csv, parse_dates=['time'])['time'].max()
    future_csv = predict_future(model_path, last_date, days)
    # 6. 옷차림 추천
    rec_path = recommend_clothing(future_csv.split('/')[-1])
    ic(f"Pipeline complete, recommendations at {rec_path}")

if __name__ == '__main__':
    fire.Fire({'run_all': run_all})
