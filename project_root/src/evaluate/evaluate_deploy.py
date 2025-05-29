import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from icecream import ic
import fire
import os
import boto3
from src.utils.utils import model_dir, ensure_dir, load_from_s3, upload_to_s3, dataset_dir
import numpy as np

def evaluate_prophet(
  bucket='mlops-weather',
  bucket_path='data/deploy_volume',
  model_name='prophet_model.pkl',
):
    key = {
      "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
      "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }
    model_path = f"{model_dir()}/{model_name}"
    testset_path = f"{dataset_dir()}/test.csv"
    # 모델 로드
    load_from_s3(bucket, bucket_path=f"{bucket_path}/model/train/{model_name}", key=key, file_path=model_path)
    model = joblib.load(model_path)
    
    # 테스트 데이터 로드
    load_from_s3(bucket, bucket_path=f"{bucket_path}/dataset/test.csv", key=key, file_path=testset_path)
    df = pd.read_csv(testset_path, parse_dates=['time'])

    test_data = df[['time', 'temp']].copy()
    test_data.columns = ['ds', 'y']
    forecast = model.predict(test_data[['ds']])
    mae = mean_absolute_error(test_data['y'], forecast['yhat'])
    rmse = root_mean_squared_error(test_data['y'], forecast['yhat'])

    ic(f"MAE: {mae}, RMSE: {rmse}")
    
if __name__ == "__main__":
    fire.Fire(evaluate_prophet)