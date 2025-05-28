from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from train import build_model

with DAG(
    'deploy_model',
    default_args={
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 0,
        'retry_delay': timedelta(minutes=5)
    },
    description='train + build_bento + build_container',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 5, 29),
    catchup=False,
    tags=['staging'],
) as dag:
    train = PythonOperator(
        task_id='training_model',
        python_callable=build_model,
    )
    build = BashOperator(
        task_id='build_bentos',
        bash_command='cd /opt/airflow/dags/service && bentoml build',
    )
    containerize = BashOperator(
        task_id='containerize_bentos',
        bash_command='cd /opt/airflow/dags/ && bentoml containerize weather_prediction_service:latest -t weather_prediction',
    )

    register_container = BashOperator(
        task_id='register_container',
        bash_command='cd /opt/airflow/dags/ && docker image tag weather_prediction:latest localhost:6000/model_serving:latest && docker push localhost:6000/model_serving:latest',
    )

train >> build >> containerize >> register_container
