from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
import os
import boto3
from airflow.models import Variable
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 경로 설정
LOCAL_CSV_PATH = "/opt/airflow/datas/tokyo_weather.csv"
PROCESSED_PATH = "/opt/airflow/datas/processed/"
PROCESSED_FILE = os.path.join(PROCESSED_PATH, "tokyo_weather_processed.csv")

# 전처리 함수 정의
def load_and_process_csv():
    print(f"✅ Loading CSV from {LOCAL_CSV_PATH}")
    df = pd.read_csv(LOCAL_CSV_PATH)
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    print(f"✅ Loading CSV from {LOCAL_CSV_PATH}")
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"✅ Processed CSV saved to {PROCESSED_FILE}")

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
    'start_date': datetime(2025, 5, 26),
    'retries': 0,  # ✅ 재시도 없음
}

with DAG(
    dag_id='csv_ingest_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='CSV 파일을 읽어 전처리 후 저장하는 DAG'
) as dag:

    csv_ingest_task = PythonOperator(
        task_id='load_and_process_csv',
        python_callable=load_and_process_csv,
    )

    upload_task = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3,
    )

    csv_ingest_task >> upload_task
