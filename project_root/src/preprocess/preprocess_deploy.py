import pandas as pd
from meteostat import Point, Hourly
from datetime import datetime, timedelta
import boto3
import os
from tqdm import tqdm
from icecream import ic
from src.utils.utils import dataset_dir, ensure_dir, load_from_s3, upload_to_s3
import fire

def preprocess_tokyo_weather(
  bucket='mlops-weather', 
  bucket_path='data/dataset/tokyo_weather_processed.csv', 
  deployment_path='data/deploy_volume', 
  test_size=0.2,
  save_name='tokyo_weather.csv'
):
    ensure_dir(dataset_dir())
    key = {
      "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
      "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }
     # 로컬에 저장할 임시 경로
    file_path = f"{dataset_dir()}/tokyo_weather.csv"

    if not os.path.exists(dataset_dir()):
        os.makedirs(dataset_dir())

    # S3에서 데이터 다운로드
    load_from_s3(bucket,bucket_path=bucket_path, key=key, file_path=file_path)
    ic(f"Downloaded weather data from {bucket}/{bucket_path} to {file_path}")

    #학습 데이터에 맞게 전처리
    df = pd.read_csv(file_path, parse_dates=['time'])
    df = df.reset_index(drop=True)
    df = df.dropna(subset=['temp'])
    df = df.sort_values('time')

    # 학습 데이터와 테스트 데이터 분리
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    train_path = f"{dataset_dir()}/train.csv"
    test_path = f"{dataset_dir()}/test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    ic(f"Train rows: {len(train)}, Test rows: {len(test)}")

    # 전처리 데이터 저장
    deploy_path_train = f"{deployment_path}/dataset/train.csv"
    deploy_path_test = f"{deployment_path}/dataset/test.csv"
    upload_to_s3(bucket, bucket_path=deploy_path_train, key=key, file_path=train_path)
    upload_to_s3(bucket, bucket_path=deploy_path_test, key=key, file_path=test_path)

if __name__ == "__main__":
    fire.Fire(preprocess_tokyo_weather)
    
