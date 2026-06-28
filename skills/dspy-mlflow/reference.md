> Condensed from [MLflow DSPy integration docs](https://mlflow.org/docs/latest/genai/flavors/dspy/index.html). Verify against upstream for latest.

# MLflow DSPy Integration — API Reference

## mlflow.dspy.autolog()

Enables automatic tracing for all DSPy calls. Must be called before any DSPy code executes.

```python
mlflow.dspy.autolog(
    log_traces=True,
    log_traces_from_compile=False,
    log_traces_from_eval=True,
    log_compiles=False,
    log_evals=False,
    disable=False,
    silent=False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_traces` | `bool` | `True` | Enable trace collection during DSPy inference |
| `log_traces_from_compile` | `bool` | `False` | Capture traces during optimizer compile() calls |
| `log_traces_from_eval` | `bool` | `True` | Capture traces when running DSPy Evaluate |
| `log_compiles` | `bool` | `False` | Record optimization metadata per compile run |
| `log_evals` | `bool` | `False` | Record evaluation call information |
| `disable` | `bool` | `False` | Turn off autologging without removing the call |
| `silent` | `bool` | `False` | Suppress MLflow event logs and warnings |

Set `log_traces_from_compile=True` to trace every LM call made during optimization — useful for debugging optimizer behavior but generates large trace volumes.

Captures: LM calls (prompt, response, tokens, latency), retrievals (query, passages), module steps (input/output), cost estimates, errors.

## mlflow.dspy.log_model()

Logs a DSPy module as an MLflow model artifact. Serializes using cloudpickle.

```python
mlflow.dspy.log_model(
    dspy_model,                      # The DSPy Module instance
    name="qa-model",                 # Artifact path name (use keyword)
    input_example=None,              # Example input for signature inference
    registered_model_name=None,      # Register directly in the registry if set
    use_dspy_model_save=False,       # Use dspy.Module.save() instead of cloudpickle
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dspy_model` | `dspy.Module` | required | The DSPy module to log |
| `name` | `str` | `None` | Artifact path name — use as keyword arg, not positional |
| `input_example` | `dict \| None` | `None` | Example input for schema inference |
| `registered_model_name` | `str \| None` | `None` | Register directly in the model registry if provided |
| `use_dspy_model_save` | `bool` | `False` | Use native `dspy.Module.save()` instead of cloudpickle serialization |

Returns model info with `model_uri` for retrieval.

## mlflow.dspy.load_model()

Loads a logged DSPy model.

```python
model = mlflow.dspy.load_model(model_uri, dst_path=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_uri` | `str` | required | MLflow model URI (e.g., `"runs:/<run_id>/model"` or `"models:/name@alias"`) |
| `dst_path` | `str \| None` | `None` | Local directory for downloaded artifacts (must exist if provided) |

Returns the native `dspy.Module` (not a PyFunc wrapper). Use `mlflow.pyfunc.load_model()` for the PyFunc-wrapped version needed by `mlflow models serve`.

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
