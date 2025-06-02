import pandas as pd
from prophet import Prophet
import joblib
from tqdm import tqdm
from icecream import ic
from src.utils.utils import model_dir, ensure_dir, load_from_s3, upload_to_s3, dataset_dir, project_path
from src.utils.modelWrapper import SarimaxWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mlflow
from mlflow import MlflowClient
import fire
import os
import boto3
import datetime

def train_sarimax(
    bucket='mlops-weather',
    bucket_path='data/deploy_volume',
    model_name='sarimax_model.pkl',
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 24),  # 시간 단위 seasonality
    enforce_stationarity=False,
    enforce_invertibility=False,
    **kwargs
):
    ensure_dir(model_dir())

    artifact_location = f"{os.getenv('MLFLOW_ARTIFACT_LOCATION')}/{os.getenv('MLFLOW_EXPERIMENT_NAME')}"

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # 실험 이름이 없으면 생성
    client = MlflowClient()
    experiment = client.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME"))

    if experiment is None:
      client.create_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"), artifact_location=artifact_location)

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

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
    df = df.sort_values("time")

    window = 24*365 # 1년
    y = df["temp"].values[-window:]

    ic(f"Training SARIMAX model with order={order} and seasonal_order={seasonal_order}")
    
    with mlflow.start_run() as run:
      mlflow.log_param("model_type", "SARIMAX")
      mlflow.log_param("order", order)
      mlflow.log_param("seasonal_order", seasonal_order)
      mlflow.log_param("enforce_stationarity", enforce_stationarity)
      mlflow.log_param("enforce_invertibility", enforce_invertibility)
      
      # 모델 학습
      model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        simple_differencing=True,
        **kwargs
      )
      
      ic("Fitting SARIMAX model...")
      results = model.fit(disp=False, low_memory=True)
      
      if not os.path.exists(model_dir()):
          os.makedirs(model_dir())

      # 모델 s3 버킷에 저장
      model_path = model_dir(model_name)
      joblib.dump(results, model_path)
      # mlflow.log_artifact(model_path)
      mlflow.pyfunc.log_model(
          artifacts={"model": model_path},
          code_path=["src"],
          artifact_path="model",  # 꼭 이 이름 써야 Registry에서 인식됨
          python_model=SarimaxWrapper(),
          # registered_model_name=f"{os.getenv('MLFLOW_EXPERIMENT_NAME')}_sarimax"  # 생략 가능, 나중에 수동 등록도 가능
      )
      ic(f"Model saved to {os.getenv('MLFLOW_ARTIFACT_LOCATION')}/{os.getenv('MLFLOW_EXPERIMENT_NAME')}")

      #run_id를 다음 컨테이너에서 사용하기 위해 파일로 저장
      run_id = run.info.run_id
      ic(f"Run ID: {run_id}")


      with open(f"{project_path()}/run_id_sarimax.txt", "w") as f:
        f.write(run_id)

      upload_to_s3(bucket, bucket_path=f"data/deploy_volume/model/{os.getenv('MLFLOW_EXPERIMENT_NAME')}/run_id_sarimax.txt", key=key, file_path=f"{project_path()}/run_id_sarimax.txt")


if __name__ == "__main__":
  fire.Fire(train_sarimax)