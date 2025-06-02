from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
from io import StringIO
import boto3
import requests
from fastapi import Request

class ModelUploadRequest(BaseModel):
    exp_name: str
    run_id: str
    pkl_file: str

key1 = os.getenv("AWS_ACCESS_KEY_ID")
key2 = os.getenv("AWS_SECRET_ACCESS_KEY")
INFERENCE_TRIGGER_URL = f"{os.getenv('INFERENCE_URL')}/run_inference"

s3 = boto3.client(
    "s3",
    aws_access_key_id=key1,
    aws_secret_access_key=key2,
    region_name="ap-northeast-2"
)

def recommend(temp):
  if temp >= 28:
      return '반팔, 반바지, 샌들 (매우 더움)'
  elif temp >= 23:
      return '반팔, 긴바지, 운동화 (더움)'
  elif temp >= 18:
      return '긴팔, 긴바지 (적당함)'
  elif temp >= 12:
      return '긴팔, 니트, 자켓 (쌀쌀함)'
  elif temp >= 5:
      return '코트, 니트, 긴바지 (추움)'
  else:
      return '두꺼운 코트, 목도리 (매우 추움)'

def get_base_df():
  try:
    response = s3.get_object(Bucket="mlops-weather", Key="data/deploy_volume/result/prediction.csv")
    data = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data))
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    return df

  except s3.exceptions.NoSuchKey:
    return None

def get_daily_df(df):
  if df is None:
    return None

  daily = df.groupby('date').agg({
        'pred_temp': ['min', 'max', 'mean']
    })
  daily.columns = ['min_temp', 'max_temp', 'avg_temp']
  daily = daily.reset_index()
  daily['clothing'] = daily['avg_temp'].apply(recommend)
  return daily


# try:
#     response = s3.get_object(Bucket="mlops-weather", Key="data/deploy_volume/result/prediction.csv")
#     data = response['Body'].read().decode('utf-8')
#     df = pd.read_csv(StringIO(data))

#     df['datetime'] = pd.to_datetime(df['datetime'])
#     df['date'] = df['datetime'].dt.date
#     daily = df.groupby('date').agg({
#         'pred_temp': ['min', 'max', 'mean']
#     })
#     daily.columns = ['min_temp', 'max_temp', 'avg_temp']
#     daily = daily.reset_index()

    
#     daily['clothing'] = daily['avg_temp'].apply(recommend)

# except s3.exceptions.NoSuchKey:
#     df = None

app = FastAPI()
app.state.df = None
app.state.daily = None

@app.get("/")
def hello():
    return {"message": "hello world 8000"}

@app.get("/result_load")
def result_load():
    app.state.df = get_base_df()
    app.state.daily = get_daily_df(app.state.df)
    if app.state.df is None:
        return {"status": "error", "message": "No base data available"}
    if app.state.daily is None:
        return {"status": "error", "message": "No daily data available"}
    return {"status": "success", 
            "df": app.state.df.to_dict(orient="records") if app.state.df is not None else None,
            "daily": app.state.daily.to_dict(orient="records") if app.state.daily is not None else None}

@app.get("/current_data")
def current_data():
    if app.state.df is None:
        return {"status": "error", "message": "No base data available"}
    return {"status": "success", "df": app.state.df.to_dict(orient="records")}

@app.get("/forecast")
def get_forecast():
    if app.state.df is None:
        app.state.df = get_base_df()
    return app.state.df.to_dict(orient="records")

@app.get("/clothing")
def get_clothing():
    if app.state.daily is None:
        if app.state.df is None:
            app.state.df = get_base_df()
        app.state.daily = get_daily_df(app.state.df)
    return app.state.daily.to_dict(orient="records")

@app.post("/model_upload")
def model_upload(request: ModelUploadRequest):
    exp_name = request.exp_name
    run_id = request.run_id
    pkl_file = request.pkl_file
    try:
        response = requests.post(INFERENCE_TRIGGER_URL, json={"exp_name": exp_name, "run_id": run_id, "pkl_file": pkl_file})
        app.state.df = get_base_df()
        app.state.daily = get_daily_df(app.state.df)
        return {
          "status": "success",
          "df": app.state.df.to_dict(orient="records") if app.state.df is not None else None,
          "daily": app.state.daily.to_dict(orient="records") if app.state.daily is not None else None,
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": 500,
            "error": str(e)
        }