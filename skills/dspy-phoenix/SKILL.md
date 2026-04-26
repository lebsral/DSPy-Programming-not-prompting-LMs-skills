---
name: dspy-phoenix
description: "Use Arize Phoenix for DSPy tracing and evaluation. Use when you want to set up Phoenix, arize-phoenix, openinference, DSPyInstrumentor, open-source trace viewer, localhost:6006, or LLM evals. Also: 'phoenix setup', 'arize phoenix', 'pip install arize-phoenix', 'phoenix local UI', 'phoenix evaluations', 'DSPy trace viewer', 'open-source LLM observability', 'phoenix vs langtrace', 'openinference-instrumentation-dspy'."
---

# Arize Phoenix — Open-Source LLM Observability for DSPy

Guide the user through setting up Arize Phoenix for DSPy tracing, visualization, and evaluation.

## What is Arize Phoenix

Phoenix is an open-source LLM observability platform that runs locally or in the cloud. It provides a trace viewer, evaluation tools, and dataset management — all with DSPy auto-instrumentation via the OpenInference plugin.

- **Local mode**: `px.launch_app()` starts a UI at `http://localhost:6006` — no account needed
- **Cloud mode**: Hosted on the [Arize platform](https://arize.com/)
- **Open source**: [github.com/Arize-ai/phoenix](https://github.com/Arize-ai/phoenix)

### What gets traced

| Component | Details captured |
|-----------|-----------------|
| LM calls | Prompts, responses, token counts, latency |
| Retrievals | Queries, passages, relevance scores |
| Module executions | Input/output per module step |
| Full pipeline | Nested spans showing the complete call tree |

## When to use Phoenix

Use Phoenix when:

- You want a local trace viewer with no cloud dependency
- You need built-in evaluation tools (evals module)
- You want an open-source solution you can self-host
- You want to visually inspect what your DSPy pipeline is doing

Do NOT use Phoenix when:

- You want the absolute easiest one-line setup — see `/dspy-langtrace`
- Your team already uses W&B — see `/dspy-weave`
- You need the full ML lifecycle (model registry, deployment) — see `/dspy-mlflow`

## Setup

### Install

```bash
pip install arize-phoenix openinference-instrumentation-dspy
```

### Local mode (quickest)

```python
import phoenix as px
from openinference.instrumentation.dspy import DSPyInstrumentor

# Launch local UI
px.launch_app()  # Opens at http://localhost:6006

# Auto-instrument DSPy
DSPyInstrumentor().instrument()

# All DSPy calls are now traced
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

program = dspy.ChainOfThought("question -> answer")
result = program(question="What is DSPy?")
# View traces at http://localhost:6006
```

### Cloud mode (Arize platform)

For teams that want persistent storage and collaboration:

```python
import phoenix as px
from openinference.instrumentation.dspy import DSPyInstrumentor

# Connect to Arize cloud
px.launch_app(endpoint="https://app.phoenix.arize.com")

DSPyInstrumentor().instrument()

# Traces are stored in the cloud — accessible to the whole team
```

### Environment variable configuration

```bash
export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006"  # or cloud URL
```

```python
import phoenix as px
from openinference.instrumentation.dspy import DSPyInstrumentor

px.launch_app()
DSPyInstrumentor().instrument()
```

## Tracing a DSPy pipeline

The OpenInference plugin auto-instruments all DSPy modules:

```python
import phoenix as px
from openinference.instrumentation.dspy import DSPyInstrumentor

px.launch_app()
DSPyInstrumentor().instrument()

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class RAGPipeline(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)

pipeline = RAGPipeline()
result = pipeline(question="How do refunds work?")
# Open http://localhost:6006 to see the trace tree:
#   RAGPipeline
#   +-- Retrieve (query, passages, latency)
#   +-- ChainOfThought (prompt, response, tokens)
```

## Inspecting traces in the Phoenix UI

The Phoenix UI at `http://localhost:6006` provides:

- **Trace list**: all requests with status, latency, and token counts
- **Trace detail**: waterfall view of every span in a request
- **Prompt viewer**: full prompt and response text for each LM call
- **Filters**: by time range, latency, status, span kind
- **Token analysis**: cost and token usage breakdowns

### Sorting and filtering

- Click column headers to sort by latency, token count, or timestamp
- Use the filter bar to find traces with specific attributes
- Click any trace to drill into the span tree

## Evaluations with Phoenix

Phoenix includes a built-in evals module for scoring LM outputs:

```python
from phoenix.evals import llm_classify, OpenAIModel

# Define evaluation criteria
eval_model = OpenAIModel(model="gpt-4o-mini")

# Score traces against criteria
eval_results = llm_classify(
    dataframe=px.Client().get_spans_dataframe(),
    model=eval_model,
    template="Is this response helpful and accurate? {output}",
    rails=["helpful", "not helpful"],
)
```

This is useful for:

- **Automated quality checks**: score every response in a batch
- **Finding failure patterns**: filter by low-scoring traces
- **Regression testing**: compare eval scores before and after changes

## Phoenix vs Langtrace vs Jaeger

| Feature | Arize Phoenix | Langtrace | Jaeger |
|---------|---------------|-----------|--------|
| DSPy auto-instrumentation | Yes (plugin) | Yes (built-in) | Manual |
| Setup effort | Two lines + launch | One line | Docker + manual spans |
| Local mode (no cloud) | Yes (`px.launch_app()`) | Yes (Docker) | Yes (Docker) |
| Cloud option | Yes (Arize platform) | Yes (app.langtrace.ai) | No |
| Built-in evals | Yes (evals module) | Basic | No |
| Dataset management | Yes | No | No |
| LM call details | Prompts, tokens, latency | Prompts, tokens, cost | Custom attributes |
| Best for | Teams wanting evals + traces | DSPy-first teams | Teams already using Jaeger |

### Decision guide

```
Want DSPy tracing?
|
+- Need built-in evals + dataset management? -> Arize Phoenix
+- Want easiest one-line setup? -> Langtrace (/dspy-langtrace)
+- Team already uses W&B? -> W&B Weave (/dspy-weave)
+- Need full ML lifecycle (registry, deploy)? -> MLflow (/dspy-mlflow)
+- Team already uses Jaeger? -> Jaeger (see /ai-tracing-requests)
```

## Cross-references

- **Langtrace** (easiest DSPy auto-instrumentation) — `/dspy-langtrace`
- **W&B Weave** (team dashboards, experiment tracking) — `/dspy-weave`
- **MLflow** (full ML lifecycle) — `/dspy-mlflow`
- **Aggregate monitoring** (not per-request) — `/ai-monitoring`
- **Per-request debugging** (inspect_history, JSONL traces) — `/ai-tracing-requests`
- For worked examples, see [examples.md](examples.md)
- Not sure which skill to use next? Try `/ai-do` to get routed to the right one
