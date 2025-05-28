import os
import random
import numpy as np

def init_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

def project_path():
    # 프로젝트 루트: utils.py에서 두 단계 위
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def dataset_dir():
    return os.path.join(project_path(), "dataset")

def model_dir(model_name=None):
    base = os.path.join(project_path(), "models")
    if model_name:
        return os.path.join(base, model_name)
    return base

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
