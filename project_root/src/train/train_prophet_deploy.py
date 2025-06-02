import pandas as pd
from prophet import Prophet
import joblib
from tqdm import tqdm
from icecream import ic
from src.utils.utils import model_dir, ensure_dir, load_from_s3, upload_to_s3, dataset_dir, project_path
import mlflow
import fire
import os
import boto3
import datetime
import mlflow.prophet
from mlflow import MlflowClient
from src.utils.modelWrapper import ProphetWrapper

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

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    artifact_location = f"{os.getenv('MLFLOW_ARTIFACT_LOCATION')}/{os.getenv('MLFLOW_EXPERIMENT_NAME')}"
    
    # 실험 이름이 없으면 생성
    client = MlflowClient()
    experiment = client.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME"))

    if experiment is None:
      client.create_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"), artifact_location=artifact_location)

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

    print('experiment_name:', os.getenv("MLFLOW_EXPERIMENT_NAME"))

    key = {
      "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
      "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }

    file_path = f"{dataset_dir()}/train.csv"
    
    if not os.path.exists(dataset_dir()):
        os.makedirs(dataset_dir())

    # 학습 데이터 로드
    load_from_s3(bucket, bucket_path=f"{bucket_path}/dataset/train.csv", key=key, file_path=file_path)
    df = pd.read_csv(file_path, parse_dates=['time'])
    data = df[['time', 'temp']].copy()
    data.columns = ['ds', 'y']
    
    with mlflow.start_run() as run:
      mlflow.log_param("model_type", "Prophet")
      mlflow.log_param("seasonality_mode", seasonality_mode)
      mlflow.log_param("daily_seasonality", daily_seasonality)
      mlflow.log_param("weekly_seasonality", weekly_seasonality)
      mlflow.log_param("yearly_seasonality", yearly_seasonality)
      mlflow.log_param("changepoint_prior_scale", changepoint_prior_scale)
      mlflow.log_param("seasonality_prior_scale", seasonality_prior_scale)

      
      # 모델 학습
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

      if not os.path.exists(model_dir()):
        os.makedirs(model_dir())

      # 모델 s3 버킷에 저장
      model_path = model_dir(model_name)
      joblib.dump(model, model_path)
      # mlflow.log_artifact(model_path)
      # log_model로 저장해야 등록 가능
      mlflow.pyfunc.log_model(
          artifacts={"model": model_path},
          python_model=ProphetWrapper(),
          code_path=["src"],
          artifact_path="model",  # 꼭 이 이름 써야 Registry에서 인식됨
          # registered_model_name=f"{os.getenv('MLFLOW_EXPERIMENT_NAME')}_prophet"  # 생략 가능, 나중에 수동 등록도 가능
      )
      ic(f"Model saved to {os.getenv('MLFLOW_ARTIFACT_LOCATION')}/{os.getenv('MLFLOW_EXPERIMENT_NAME')}")

      #run_id를 다음 컨테이너에서 사용하기 위해 파일로 저장
      run_id = run.info.run_id
      ic(f"Run ID: {run_id}")


      with open(f"{project_path()}/run_id_prophet.txt", "w") as f:
        f.write(run_id)

      upload_to_s3(bucket, bucket_path=f"data/deploy_volume/model/{os.getenv('MLFLOW_EXPERIMENT_NAME')}/run_id_prophet.txt", key=key, file_path=f"{project_path()}/run_id_prophet.txt")

      

if __name__ == "__main__":
  fire.Fire(train_prophet)