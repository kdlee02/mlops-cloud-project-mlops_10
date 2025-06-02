import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from icecream import ic
import fire
import os
import boto3
from src.utils.utils import model_dir, ensure_dir, load_from_s3, upload_to_s3, dataset_dir, project_path
import numpy as np
import mlflow


def evaluate_sarimax(
  bucket='mlops-weather',
  bucket_path='data/deploy_volume',
  model_name='sarimax_model.pkl',
):
    key = {
      "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
      "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }
    run_id_path = f"{project_path()}/run_id_sarimax.txt"
    model_path = f"{model_dir()}/{model_name}"
    testset_path = f"{dataset_dir()}/test.csv"

    if not os.path.exists(model_dir()):
        os.makedirs(model_dir())

    if not os.path.exists(dataset_dir()):
        os.makedirs(dataset_dir())

    #s3에서 모델 등록된 위치 찾기
    run_id_bucket_path = f"{bucket_path}/model/{os.getenv('MLFLOW_EXPERIMENT_NAME')}/run_id_sarimax.txt"
    load_from_s3(
      bucket, 
      bucket_path=run_id_bucket_path, 
      key=key, 
      file_path=run_id_path
    )

    with open(run_id_path, "r") as f:
      run_id = f.read()

    ic(f"run_id: {run_id}")
    bucket_model_path = f"{bucket_path}/model/{os.getenv('MLFLOW_EXPERIMENT_NAME')}/{run_id}/artifacts/model/artifacts/{model_name}"

    # 모델 로드
    load_from_s3(bucket, bucket_path=bucket_model_path, key=key, file_path=model_path)
    model = joblib.load(model_path)
    
    # 테스트 데이터 로드
    load_from_s3(bucket, bucket_path=f"{bucket_path}/dataset/test.csv", key=key, file_path=testset_path)
    df = pd.read_csv(testset_path, parse_dates=['time'])

    y_true = df['temp'].values
    start = 0
    end = len(y_true) - 1
    y_pred = model.predict(start=start, end=end)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    ic(f"MAE: {mae}, RMSE: {rmse}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
    
if __name__ == "__main__":
    fire.Fire(evaluate_sarimax)