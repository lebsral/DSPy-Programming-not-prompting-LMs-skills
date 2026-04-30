> Condensed from [MLflow DSPy integration docs](https://mlflow.org/docs/latest/genai/flavors/dspy/index.html). Verify against upstream for latest.

# MLflow DSPy Integration — API Reference

## mlflow.dspy.autolog()

Enables automatic tracing for all DSPy calls. Must be called before any DSPy code executes.

```python
mlflow.dspy.autolog()
```

Captures: LM calls (prompt, response, tokens, latency), retrievals (query, passages), module steps (input/output), cost estimates, errors.

## mlflow.dspy.log_model()

Logs a DSPy module as an MLflow model artifact. Serializes using cloudpickle.

```python
mlflow.dspy.log_model(
    dspy_model,      # The DSPy Module instance
    name,            # Artifact name (e.g., "qa-model")
    input_example=None,  # Example input for signature inference
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dspy_model` | `dspy.Module` | The DSPy module to log |
| `name` | `str` | Artifact path name |
| `input_example` | `dict \| None` | Optional example input for schema inference |

Returns model info with `model_uri` for retrieval.

## mlflow.dspy.load_model()

Loads a logged DSPy model.

```python
model = mlflow.dspy.load_model(model_uri)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_uri` | `str` | MLflow model URI (e.g., `"runs:/<run_id>/model"` or `"models:/name@alias"`) |

Returns the native DSPy module (not a PyFunc wrapper).

## mlflow.dspy.save_model()

Saves a DSPy module to a local directory (alternative to `log_model` for offline use).

```python
mlflow.dspy.save_model(dspy_model, path)
```

## Model aliases (replaces deprecated model stages)

```python
from mlflow import MlflowClient
client = MlflowClient()

# Set alias
client.set_registered_model_alias("model-name", "champion", version=2)

# Load by alias
model = mlflow.dspy.load_model("models:/model-name@champion")

# Delete alias
client.delete_registered_model_alias("model-name", "champion")
```

## Experiment tracking

```python
mlflow.set_experiment("experiment-name")

with mlflow.start_run(run_name="run-name"):
    mlflow.log_param("key", "value")
    mlflow.log_metric("score", 0.85)
    mlflow.log_artifact("file.json")
```

| Function | Description |
|----------|-------------|
| `mlflow.set_experiment(name)` | Set active experiment (creates if needed) |
| `mlflow.start_run(run_name=None)` | Start a tracked run (use as context manager) |
| `mlflow.log_param(key, value)` | Log a parameter (string) |
| `mlflow.log_metric(key, value)` | Log a metric (numeric) |
| `mlflow.log_artifact(path)` | Log a file artifact |
| `mlflow.register_model(model_uri, name)` | Register model in registry |

## Key limitations

- API tokens and secrets are not serialized — set via environment variables in production
- DSPy trace objects are not serializable
- MLflow DSPy integration is marked as experimental — APIs may change
