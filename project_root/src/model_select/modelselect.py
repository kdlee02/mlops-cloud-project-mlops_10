import mlflow
from mlflow.tracking import MlflowClient

def get_best_model(experiment_name: str, metric: str = "mse"):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found")

    model_uri = f"runs:/{runs[0].info.run_id}/model"

    model = mlflow.pyfunc.load_model(model_uri)

    return model