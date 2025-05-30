from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models import Variable
from datetime import datetime, timedelta

with DAG(
    dag_id='ml_pipeline',
    start_date=datetime.now() - timedelta(days=1),  # ✅ 과거 날짜로 설정
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 's3'],
) as dag:

    preprocess = DockerOperator(
        task_id='preprocess_data',
        image='jbreal/mlops-deployment:latest',
        api_version='auto',
        auto_remove=True,
        environment={
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY"),
            "AWS_DEFAULT_REGION": "ap-northeast-2"
        },
        command='python src/preprocess/preprocess_deploy.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge'
    )

    train = DockerOperator(
        task_id='train_data',
        image='jbreal/mlops-deployment:latest',
        api_version='auto',
        auto_remove=True,
        environment={
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY"),
            "AWS_DEFAULT_REGION": "ap-northeast-2"
        },
        command='python src/train/train_deploy.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge'
    )

    evaluate = DockerOperator(
        task_id='evaluate_data',
        image='jbreal/mlops-deployment:latest',
        api_version='auto',
        auto_remove=True,
        environment={
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY"),
            "AWS_DEFAULT_REGION": "ap-northeast-2"
        },
        command='python src/evaluate/evaluate_deploy.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge'
    )

    preprocess >> train >> evaluate

