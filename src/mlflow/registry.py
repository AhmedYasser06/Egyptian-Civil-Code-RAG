"""Register the winning chunking configuration in the MLflow Model Registry and promote it.

The "model" is a tiny pyfunc that returns the chosen chunking/retrieval config, so the registry
holds the production decision (chunk_size, overlap, strategy, embedding model, ...) next to the
metrics that justified it.
"""

from __future__ import annotations

import json

import mlflow
from mlflow import MlflowClient

REGISTERED_MODEL = "civil-code-rag-chunking"


class ChunkingConfigModel(mlflow.pyfunc.PythonModel):
    def __init__(self, config: dict):
        self.config = config

    def predict(self, context, model_input, params=None):
        return [json.dumps(self.config)] * len(model_input)


def register_best(
    config: dict,
    metrics: dict,
    run_name: str = "register-best-chunking",
    name: str = REGISTERED_MODEL,
) -> dict:
    """Log the config as a pyfunc model, register it, and promote it to Production.

    MLflow 3 deprecates stages in favour of aliases, so both are set: the ``production`` alias
    (what code should load: ``models:/civil-code-rag-chunking@production``) and the legacy
    ``Production`` stage that the course rubric asks for (best effort).
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({f"best_{k}": v for k, v in config.items()})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        info = mlflow.pyfunc.log_model(
            name="chunking_config",
            python_model=ChunkingConfigModel(config),
            pip_requirements=["mlflow"],
            registered_model_name=name,
        )
    client = MlflowClient()
    version = str(info.registered_model_version)
    client.set_registered_model_alias(name, "production", version)
    client.set_model_version_tag(name, version, "chosen_by", "highest RAGAS faithfulness")
    stage_set = False
    try:
        client.transition_model_version_stage(
            name, version, "Production", archive_existing_versions=True
        )
        stage_set = True
    except Exception as exc:  # removed/deprecated in newer MLflow: the alias is authoritative
        print(f"[registry] stage transition skipped ({type(exc).__name__}); alias 'production' set")
    return {"name": name, "version": version, "run_id": run.info.run_id, "stage_set": stage_set}
