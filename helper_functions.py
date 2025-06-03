import os
import logging
import pickle

def log_info(message):
    """Log informational messages"""
    logging.info(message)
    print(f"[INFO] {message}")

def setup_logging(log_dir="logs"):
    """Configure logging system"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "lung_cancer_mlops.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    log_info("Logging system initialized")

def save_artifact(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure parent directory exists
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    log_info(f"Artifact saved: {file_path}")


def load_artifact(file_path):
    """Load object from pickle file"""
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    log_info(f"Artifact loaded: {file_path}")
    return obj

def get_env_variable(name, default=None):
    """Get environment variable with fallback"""
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable {name} not set")
    return value