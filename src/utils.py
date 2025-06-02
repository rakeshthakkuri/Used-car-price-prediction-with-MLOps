import os
import json
import pickle

def save_pickle(obj, path):
    """
    Save a Python object to a pickle file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    """
    Load a Python object from a pickle file.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_json(data, path):
    """
    Save dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    """
    Load dictionary from a JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)

def print_metrics(metrics_dict):
    """
    Nicely print evaluation metrics.
    """
    print("Evaluation Metrics:")
    for key, value in metrics_dict.items():
        print(f"{key}: {value:.4f}")
