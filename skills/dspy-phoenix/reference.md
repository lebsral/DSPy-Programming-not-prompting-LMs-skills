> Condensed from [Phoenix DSPy docs](https://arize.com/docs/phoenix/integrations/python/dspy/dspy-tracing) and [Phoenix quickstart](https://arize.com/docs/phoenix/quickstart). Verify against upstream for latest.

# Arize Phoenix — API Reference for DSPy

## Installation

```bash
pip install arize-phoenix openinference-instrumentation-dspy openinference-instrumentation-litellm
```

| Package | Purpose |
|---------|---------|
| `arize-phoenix` | Phoenix server, UI, evals module, client |
| `openinference-instrumentation-dspy` | Auto-instruments DSPy modules (Predict, ChainOfThought, etc.) |
| `openinference-instrumentation-litellm` | Instruments LiteLLM (used by DSPy under the hood) for token counts and costs |

## Core setup functions

### `px.launch_app()`

Starts a local Phoenix server with UI.

```python
import phoenix as px

session = px.launch_app()  # UI at http://localhost:6006
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port` | `int` | `6006` | Port for the local UI |

### `phoenix.otel.register()`

Registers a tracer provider with auto-instrumentation.

```python
from phoenix.otel import register

tracer_provider = register(
    project_name="my-project",
    auto_instrument=True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_name` | `str` | `"default"` | Project name shown in Phoenix UI |
| `auto_instrument` | `bool` | `False` | Auto-discovers and instruments all installed OpenInference packages |

### `phoenix.otel.using_attributes()`

Context manager to attach metadata to traces.

```python
from phoenix.otel import using_attributes

with using_attributes(
    session_id="session-123",
    user_id="user-456",
    metadata={"env": "prod"},
    tags=["experiment-v2"],
    prompt_template_version="v1.2",
    prompt_template_variables={"tone": "formal"},
):
    result = program(question="...")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Groups traces by session |
| `user_id` | `str` | Associates traces with a user |
| `metadata` | `dict` | Arbitrary key-value metadata |
| `tags` | `list[str]` | Searchable tags |
| `prompt_template_version` | `str` | Version identifier for prompt template |
| `prompt_template_variables` | `dict` | Variables used in the prompt template |

## Environment variables

| Variable | Description |
|----------|-------------|
| `PHOENIX_COLLECTOR_ENDPOINT` | URL where traces are sent (default: `http://localhost:6006`) |
| `PHOENIX_API_KEY` | API key for Arize cloud mode |

## Phoenix Client

```python
client = px.Client()
```

| Method | Returns | Description |
|--------|---------|-------------|
| `get_spans_dataframe()` | `pd.DataFrame` | All spans as a DataFrame (for evals) |
| `get_trace_dataset()` | `TraceDataset` | Traces as a dataset object |

## Evals module

```python
from phoenix.evals import llm_classify, OpenAIModel

eval_model = OpenAIModel(model="gpt-4o-mini")

results = llm_classify(
    dataframe=spans_df,
    model=eval_model,
    template="Is this response helpful? {output}",
    rails=["helpful", "not helpful"],
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataframe` | `pd.DataFrame` | Spans to evaluate (from `px.Client().get_spans_dataframe()`) |
| `model` | `BaseModel` | LM to use for evaluation |
| `template` | `str` | Prompt template with `{column_name}` placeholders |
| `rails` | `list[str]` | Allowed classification labels |

## Legacy pattern (still works but not recommended)

```python
# Old pattern — instruments DSPy only, misses LiteLLM spans
from openinference.instrumentation.dspy import DSPyInstrumentor
DSPyInstrumentor().instrument()

# New pattern — auto-instruments all installed OpenInference packages
from phoenix.otel import register
register(project_name="my-project", auto_instrument=True)
```
