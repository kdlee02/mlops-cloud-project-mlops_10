import pandas as pd
from prophet import Prophet
import joblib
from tqdm import tqdm
from icecream import ic
from src.utils.utils import model_dir, ensure_dir, load_from_s3, upload_to_s3, dataset_dir
import fire
import os
import boto3

def train_prophet(
    bucket='mlops-weather',
    bucket_path='data/deploy_volume',
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

    key = {
      "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
      "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }

    file_path = f"{dataset_dir()}/train.csv"

    # 학습 데이터 로드
    load_from_s3(bucket, bucket_path=f"{bucket_path}/dataset/train.csv", key=key, file_path=file_path)
    df = pd.read_csv(file_path, parse_dates=['time'])

    # 모델 학습
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

    # 모델 저장
    model_path = model_dir(model_name)
    joblib.dump(model, model_path)
    upload_to_s3(bucket, bucket_path=f"{bucket_path}/model/train/{model_name}", key=key, file_path=model_path)
    ic(f"Model saved to s3://{bucket}/{bucket_path}/model/train/{model_name}")

if __name__ == "__main__":
  fire.Fire(train_prophet)