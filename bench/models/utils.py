import os

def ensure_models_dir():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

