import os
import sys

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from evaluate import evaluate_model

def run_pipeline_evaluation(model, X_test, y_test):
    print("Starting evaluation...")
    evaluate_model(model, X_test, y_test)
    print("Evaluation completed.")

if __name__ == "__main__":
    # Example: If run standalone, can load model & test data from saved files or pipeline args
    # For demo, raise error to force explicit use
    raise NotImplementedError("Run evaluation with explicit model, X_test, y_test")
