from fastapi import FastAPI
import os
import boto3
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI()

# 🔐 환경 변수 기반
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = "ap-northeast-2"

BUCKET_NAME = "mlops-weather"
MODEL_S3_KEY = "data/deploy_volume/model/train/prophet_model.pkl"
RESULT_S3_KEY = "data/deploy_volume/result/prediction.csv"
LOCAL_MODEL_PATH = os.path.join("model", "prophet_model.pkl")
LOCAL_RESULT_PATH = os.path.join("result", "prediction.csv")

os.makedirs("model", exist_ok=True)
os.makedirs("result", exist_ok=True)

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/run_inference")
def run_inference():
    try:
        # 1. 모델 다운로드
        s3.download_file(BUCKET_NAME, MODEL_S3_KEY, LOCAL_MODEL_PATH)

        # 2. 추론
        model = joblib.load(LOCAL_MODEL_PATH)
        future = pd.date_range(start=pd.Timestamp.now(), periods=24, freq="H")
        df_future = pd.DataFrame({"ds": future})
        forecast = model.predict(df_future)
        result = forecast[["ds", "yhat"]].copy()
        result.columns = ["datetime", "pred_temp"]

        # 3. 로컬 저장
        result.to_csv(LOCAL_RESULT_PATH, index=False)

        # 4. 결과 업로드
        s3.upload_file(LOCAL_RESULT_PATH, BUCKET_NAME, RESULT_S3_KEY)

        return {
            "status": "success",
            "rows": len(result),
            "s3_result_key": RESULT_S3_KEY
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}