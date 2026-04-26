---
name: dspy-langtrace
description: Use Langtrace for DSPy observability and tracing. Use when you want to set up Langtrace, langtrace-python-sdk, auto-instrument DSPy, trace DSPy calls, LLM observability, app.langtrace.ai, or self-hosted tracing. Also: 'langtrace setup', 'langtrace API key', 'pip install langtrace-python-sdk', 'DSPy tracing', 'auto-instrument DSPy', 'langtrace self-hosted', 'langtrace docker', 'trace LM calls', 'langtrace vs phoenix', 'langtrace cloud'.
---

# Langtrace — Open-Source LLM Observability for DSPy

Guide the user through setting up Langtrace for automatic DSPy tracing and observability.

## What is Langtrace

Langtrace is an open-source LLM observability platform with **first-class DSPy auto-instrumentation**. One line of code traces all DSPy LM calls, retrievals, module executions, token counts, and cost — no manual decorators needed.

- **Cloud**: Managed at [app.langtrace.ai](https://app.langtrace.ai)
- **Self-hosted**: Run your own instance with Docker
- **Open source**: [github.com/Scale3-Labs/langtrace](https://github.com/Scale3-Labs/langtrace)

### What gets traced automatically

| Component | Details captured |
|-----------|-----------------|
| LM calls | Prompts, responses, token counts, cost, latency |
| Retrievals | Queries, retrieved passages, scores |
| Module executions | Input/output per `dspy.Module.forward()` call |
| Nested pipelines | Full call tree with parent-child relationships |

## When to use Langtrace

Use Langtrace when:

- You want the easiest DSPy tracing setup (one line)
- You need auto-instrumentation without decorating every function
- You want a cloud dashboard with no infrastructure to manage
- You need self-hosted tracing for data privacy

Do NOT use Langtrace when:

- You need deep evaluation/evals features — see `/dspy-phoenix` (Phoenix has built-in evals)
- Your team is already invested in W&B for experiment tracking — see `/dspy-weave`
- You need the full ML lifecycle (model registry, deployment) — see `/dspy-mlflow`

## Setup

### Install

```bash
pip install langtrace-python-sdk
```

### Cloud setup (quickest)

1. Sign up at [app.langtrace.ai](https://app.langtrace.ai)
2. Create a project and copy your API key
3. Add two lines to your code:

```python
from langtrace_python_sdk import langtrace

langtrace.init(api_key="your-key")  # or set LANGTRACE_API_KEY env var

# That's it — all DSPy calls are now traced automatically
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

program = dspy.ChainOfThought("question -> answer")
result = program(question="What is DSPy?")
# View traces at app.langtrace.ai
```

### Self-hosted setup (Docker)

For teams that need data to stay on-premises:

```bash
# Clone and start Langtrace
git clone https://github.com/Scale3-Labs/langtrace.git
cd langtrace
docker compose up -d
```

Then point your SDK at your local instance:

```python
from langtrace_python_sdk import langtrace

langtrace.init(api_host="http://localhost:3000")

# All traces go to your self-hosted instance
```

### Environment variable configuration

```bash
export LANGTRACE_API_KEY="your-key"           # Cloud API key
# OR
export LANGTRACE_API_HOST="http://localhost:3000"  # Self-hosted URL
```

```python
from langtrace_python_sdk import langtrace

langtrace.init()  # Picks up from environment variables
```

## Tracing a DSPy pipeline

Langtrace auto-instruments the entire call tree. No changes to your DSPy code:

```python
from langtrace_python_sdk import langtrace

langtrace.init(api_key="your-key")

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
# Langtrace captures:
#   - The top-level RAGPipeline call
#   - The Retrieve call (query, passages, latency)
#   - The ChainOfThought LM call (prompt, response, tokens, cost)
```

## Tracing optimization runs

Langtrace traces optimizer internals too — useful for understanding what MIPROv2 or GEPA tried:

```python
from langtrace_python_sdk import langtrace

langtrace.init(api_key="your-key")

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

trainset = [...]  # your training examples

program = dspy.ChainOfThought("question -> answer")
optimizer = dspy.MIPROv2(metric=my_metric, auto="light")
optimized = optimizer.compile(program, trainset=trainset)
# Every LM call the optimizer makes is traced — see which candidates it tried
```

## Viewing traces in the Langtrace UI

The Langtrace dashboard shows:

- **Trace timeline**: waterfall view of every step in a request
- **Token counts & cost**: per-call and aggregate
- **Latency breakdown**: which step is slowest
- **Prompt/response viewer**: full text of every LM interaction
- **Filters**: by time range, latency, status, and custom attributes

### Adding custom attributes

Tag traces with metadata for filtering:

```python
from langtrace_python_sdk import langtrace, with_langtrace_root_span

@with_langtrace_root_span("customer-query")
def handle_query(user_id, question):
    # Custom attributes appear in the UI for filtering
    langtrace.inject_additional_attributes({
        "user_id": user_id,
        "environment": "production",
    })
    return pipeline(question=question)
```

## Langtrace vs Phoenix vs Jaeger

| Feature | Langtrace | Arize Phoenix | Jaeger |
|---------|-----------|---------------|--------|
| DSPy auto-instrumentation | Yes (built-in) | Yes (plugin) | Manual |
| Setup effort | One line | Two lines + launch | Docker + manual spans |
| Self-hosted option | Yes (Docker) | Yes | Yes |
| Cloud option | Yes (app.langtrace.ai) | Yes (Arize platform) | No |
| LM call details | Prompts, tokens, cost | Prompts, tokens | Custom attributes |
| Evals/evaluation | Basic | Built-in evals module | No |
| Best for | DSPy-first teams | Teams wanting evals + traces | Teams already using Jaeger |

### Decision guide

```
Want DSPy tracing?
|
+- Easiest setup, auto-instrument everything? -> Langtrace
+- Need built-in evaluation features? -> Arize Phoenix (/dspy-phoenix)
+- Team already uses W&B? -> W&B Weave (/dspy-weave)
+- Need full ML lifecycle (registry, deploy)? -> MLflow (/dspy-mlflow)
+- Team already uses Jaeger? -> Jaeger (see /ai-tracing-requests)
```

## Cross-references

- **Arize Phoenix** (open-source with evals) — `/dspy-phoenix`
- **W&B Weave** (team dashboards, experiment tracking) — `/dspy-weave`
- **MLflow** (full ML lifecycle) — `/dspy-mlflow`
- **Aggregate monitoring** (not per-request) — `/ai-monitoring`
- **Per-request debugging** (inspect_history, JSONL traces) — `/ai-tracing-requests`
- For worked examples, see [examples.md](examples.md)
- Not sure which skill to use next? Try `/ai-do` to get routed to the right one
