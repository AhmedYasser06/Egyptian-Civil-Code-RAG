import mlflow

EXPERIMENT_NAME = "Egyptian-Civil-Code-RAG"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    print(f"Created experiment: {EXPERIMENT_NAME}")
    print(f"Experiment ID: {experiment_id}")
else:
    print(f"Experiment already exists: {EXPERIMENT_NAME}")
    print(f"Experiment ID: {experiment.experiment_id}")

print(f"Tracking URI: {mlflow.get_tracking_uri()}")
