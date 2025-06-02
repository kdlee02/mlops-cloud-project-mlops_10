from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import os
from io import StringIO
import boto3


key1 = os.getenv("AWS_ACCESS_KEY_ID")
key2 = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    aws_access_key_id=key1,
    aws_secret_access_key=key2,
    region_name="ap-northeast-2"
)

try:
    response = s3.get_object(Bucket="mlops-weather", Key="data/deploy_volume/result/prediction.csv")
    data = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data))

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    daily = df.groupby('date').agg({
        'pred_temp': ['min', 'max', 'mean']
    })
    daily.columns = ['min_temp', 'max_temp', 'avg_temp']
    daily = daily.reset_index()

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
    daily['clothing'] = daily['avg_temp'].apply(recommend)

except s3.exceptions.NoSuchKey:
    df = None

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "hello world 8000"}

@app.get("/forecast")
def get_forecast():
    return df.to_dict(orient="records")

@app.get("/clothing")
def get_clothing():
    return daily.to_dict(orient="records")
