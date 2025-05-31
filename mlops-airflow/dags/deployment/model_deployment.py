import sys
import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import mlflow

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from plugins.utils import get_next_deployment_experiment_name


mlflow_uri = "http://mlflow-server:5000"
mlflow.set_tracking_uri(mlflow_uri)

experiment_name = get_next_deployment_experiment_name()

with DAG(
    dag_id='ml_pipeline',
    start_date=datetime.now() - timedelta(days=1),
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 's3'],
) as dag:

    COMMON_ENV = {
      "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID"),
      "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY"),
      "AWS_DEFAULT_REGION": "ap-northeast-2",
      "MLFLOW_TRACKING_URI": mlflow_uri,
      "MLFLOW_EXPERIMENT_NAME": experiment_name,
      "MLFLOW_ARTIFACT_LOCATION": f"s3://mlops-weather/data/deploy_volume/model",
    }
    
    preprocess = DockerOperator(
        task_id='preprocess_data',
        image='jbreal/mlops-deployment:latest',
        api_version='auto',
        auto_remove=True,
        environment=COMMON_ENV,
        command='python src/preprocess/preprocess_deploy.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops-net'
    )

    train_prophet = DockerOperator(
        task_id='train_prophet',
        image='jbreal/mlops-deployment:latest',
        api_version='auto',
        auto_remove=True,
        environment=COMMON_ENV,
        command='python src/train/train_prophet_deploy.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops-net'
    )

    train_sarimax = DockerOperator(
        task_id='train_sarimax',
        image='jbreal/mlops-deployment:latest',
        api_version='auto',
        auto_remove=True,
        environment=COMMON_ENV,
        command='python src/train/train_sarimax_deploy.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops-net'
    )

    evaluate_prophet = DockerOperator(
        task_id='evaluate_prophet',
        image='jbreal/mlops-deployment:latest',
        api_version='auto',
        auto_remove=True,
        environment=COMMON_ENV,
        command='python src/evaluate/evaluate_prophet_deploy.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops-net'
    )

    evaluate_sarimax = DockerOperator(
        task_id='evaluate_sarimax',
        image='jbreal/mlops-deployment:latest',
        api_version='auto',
        auto_remove=True,
        environment=COMMON_ENV,
        command='python src/evaluate/evaluate_sarimax_deploy.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops-net'
    )

    preprocess >> train_prophet >> evaluate_prophet
    preprocess >> train_sarimax >> evaluate_sarimax

