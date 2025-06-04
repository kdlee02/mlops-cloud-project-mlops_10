from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from meteostat import Point, Hourly
from datetime import datetime, timedelta
import pandas as pd
import os
import boto3
from airflow.models import Variable
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 경로 설정
PROCESSED_PATH = "/opt/airflow/datas"
PROCESSED_FILE = os.path.join(PROCESSED_PATH, "tokyo_weather_processed.csv")

def collect_tokyo_weather():
    tokyo = Point(35.6762, 139.6503, 70)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * 3)
    data = Hourly(tokyo, start_date, end_date).fetch()
    df = data.reset_index()
    print(len(df))
    df.to_csv(PROCESSED_FILE, index=False)
    

# S3 업로드 함수 정의
def upload_to_s3():
    s3 = boto3.client(
        's3',
        aws_access_key_id=Variable.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=Variable.get("AWS_SECRET_ACCESS_KEY"),
        region_name="ap-northeast-2"  # 서울 리전 등으로 설정
    )
    bucket_name = "mlops-weather"
    object_name = "data/dataset/tokyo_weather_processed.csv"

    with open(PROCESSED_FILE, "rb") as f:
        s3.upload_fileobj(f, bucket_name, object_name)

    print(f"✅ Uploaded {PROCESSED_FILE} to s3://{bucket_name}/{object_name}")    

# DAG 정의
default_args = {
    'owner': 'airflow',
    'start_date': datetime.now() - timedelta(days=1),
    'retries': 0,  # ✅ 재시도 없음
}

with DAG(
    dag_id='weather_data_collection_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    description='Meteostat를 사용하여 도쿄 날씨 데이터를 수집하고 CSV 파일로 s3에 저장하는 DAG'
) as dag:

    collect_weather_task = PythonOperator(
        task_id='collect_tokyo_weather',
        python_callable=collect_tokyo_weather,
    )

    upload_task = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3,
    )

    collect_weather_task >> upload_task
