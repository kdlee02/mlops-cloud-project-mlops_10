import re
from mlflow import MlflowClient

def get_next_deployment_experiment_name(base_name="deployment"):
    client = MlflowClient()
    # list_experiments() → search_experiments()
    experiments = client.search_experiments()

    pattern = re.compile(f"^{base_name}-(\\d+)$")
    max_id = 0
    found = False

    for exp in experiments:
        match = pattern.match(exp.name)
        if match:
            found = True
            num = int(match.group(1))
            max_id = max(max_id, num)

    if not found:
        return f"{base_name}-1"
    else:
        return f"{base_name}-{max_id + 1}"
