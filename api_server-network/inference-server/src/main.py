from fastapi import FastAPI
import mlflow
import shutil
import os
import boto3
from mlflow import MlflowClient

app = FastAPI()

# MODEL_NAME = "Prophet_deploy-v12"
# LOCAL_MODEL_PATH = "/app/model/model.pkl"
# MLFLOW_TRACKING_URI = "http://mlflow-server:5000"

@app.get("/")
def hello():
    return {"message": "hello world 8001"}

# @app.get("/get_model")
# def get_latest_model():
#     try:
#         mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#         client = MlflowClient()

#         latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
#         if not latest_versions:
#             return {"status": "error", "message": "No production model found"}

#         run_id = latest_versions[0].run_id
#         run_info = client.get_run(run_id).info
#         artifact_uri = run_info.artifact_uri  # 예: s3://mlops-weather/...

#         # artifact_uri에서 S3 bucket과 key 추출
#         parsed = urlparse(artifact_uri)
#         if parsed.scheme != "s3":
#             return {"status": "error", "message": f"Unsupported scheme: {parsed.scheme}"}

#         bucket = parsed.netloc
#         key_prefix = parsed.path.lstrip("/")  # artifacts 경로
#         model_key = os.path.join(key_prefix, "model", "model.pkl")
#         local_path = os.path.join("model", "model.pkl")
#         os.makedirs("model", exist_ok=True)

#         # S3 다운로드
#         s3 = boto3.client("s3",
#             aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#             aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#             region_name="ap-northeast-2"
#         )
#         s3.download_file(bucket, model_key, local_path)

#         return {
#             "status": "ok",
#             "run_id": run_id,
#             "local_model_path": local_path
#         }

#     except Exception as e:
#         return {"status": "error", "message": str(e)}

