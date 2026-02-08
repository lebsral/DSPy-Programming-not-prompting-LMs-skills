---
name: ai-tracing-requests
description: See exactly what your AI did on a specific request. Use when you need to debug a wrong answer, trace a specific AI request, profile slow AI pipelines, find which step failed, inspect LM calls, view token usage per request, build audit trails, or understand why a customer got a bad response. Covers DSPy inspection, per-step tracing, OpenTelemetry instrumentation, and trace viewer setup.
---

# See What Your AI Did on a Specific Request

Guide the user through tracing and debugging individual AI requests. The goal: for any request, see every LM call, retrieval step, intermediate result, token count, and latency.

## When you need this

- A customer reports a wrong answer — you need to see exactly what happened
- Your pipeline is slow — you need to find which step is the bottleneck
- Compliance requires audit trails of every AI decision
- QA wants to inspect AI behavior before launch
- You're debugging why an agent took unexpected actions

## How it's different from monitoring

| | Monitoring (`/ai-monitoring`) | Tracing (this skill) |
|---|---|---|
| Scope | Aggregate health across all requests | Single request, full detail |
| Question answered | "Is accuracy dropping this week?" | "Why did customer #12345 get a wrong answer at 2:14pm?" |
| Output | Scores, trends, alerts | Call traces, intermediate results, latencies |
| Timing | Periodic batch evaluation | Per-request, real-time |

## Step 1: Understand the need

Quick decision tree:

```
What are you debugging?
|
+- A specific wrong answer right now?
|  -> Step 2: Quick debugging with dspy.inspect_history
|
+- Need to trace requests in a running app?
|  -> Step 3-4: Add per-step tracing
|
+- Need a visual trace viewer for your team?
|  -> Step 5: Connect Langtrace, Phoenix, or Jaeger
|
+- Need to find patterns across many traces?
   -> Step 6: Search and filter traces
```

## Step 2: Quick debugging (no extra tools needed)

### Inspect the last LM calls

The fastest way to see what happened:

```python
import dspy

# Run your program
result = my_program(question="What is our refund policy?")

# See the last 5 LM calls — shows full prompts and responses
dspy.inspect_history(n=5)
```

This shows:
- The full prompt sent to the LM (including system message, few-shot examples, input)
- The LM's raw response
- How DSPy parsed the response into fields

### Time individual steps

```python
import time

result = my_program(question="test")

# Quick manual timing
start = time.time()
step1_result = my_program.step1(question="test")
step1_time = time.time() - start
print(f"Step 1: {step1_time:.2f}s")

start = time.time()
step2_result = my_program.step2(context=step1_result.context, question="test")
step2_time = time.time() - start
print(f"Step 2: {step2_time:.2f}s")
```

### JSONL trace logging

For persistent traces without any extra dependencies:

```python
import json
import time
from datetime import datetime

class TracedProgram(dspy.Module):
    """Wraps any DSPy program to log per-step traces to JSONL."""
    def __init__(self, program, log_path="traces.jsonl"):
        self.program = program
        self.log_path = log_path

    def forward(self, **kwargs):
        trace_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        steps = []

        start = time.time()
        result = self.program(**kwargs)
        total_time = time.time() - start

        # Log the trace
        entry = {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "inputs": {k: str(v) for k, v in kwargs.items()},
            "outputs": {k: str(getattr(result, k, "")) for k in result.keys()},
            "total_latency_ms": round(total_time * 1000),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return result

# Use it
traced = TracedProgram(my_program)
result = traced(question="How do refunds work?")
```

## Step 3: Per-step tracing in pipelines

For multi-step pipelines, trace each stage separately to see exactly where things go wrong:

```python
import json
import time
import uuid
from datetime import datetime

class StepTracer:
    """Collects per-step timing and intermediate results."""
    def __init__(self):
        self.steps = []
        self.trace_id = str(uuid.uuid4())[:8]

    def trace_step(self, name, func, **kwargs):
        """Run a step and record its inputs, outputs, and latency."""
        start = time.time()
        result = func(**kwargs)
        latency = time.time() - start

        self.steps.append({
            "step": name,
            "inputs": {k: str(v)[:200] for k, v in kwargs.items()},
            "outputs": {k: str(getattr(result, k, ""))[:200] for k in result.keys()},
            "latency_ms": round(latency * 1000),
        })
        return result

    def summary(self):
        """Print a summary of all traced steps."""
        print(f"Trace {self.trace_id}:")
        total = sum(s["latency_ms"] for s in self.steps)
        for step in self.steps:
            pct = step["latency_ms"] / total * 100 if total > 0 else 0
            print(f"  {step['step']}: {step['latency_ms']}ms ({pct:.0f}%)")
        print(f"  Total: {total}ms")

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "timestamp": datetime.now().isoformat(),
            "steps": self.steps,
            "total_latency_ms": sum(s["latency_ms"] for s in self.steps),
        }

# Use in a pipeline
class TracedRAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        tracer = StepTracer()

        retrieval = tracer.trace_step("retrieve", self.retrieve, query=question)

        answer = tracer.trace_step(
            "answer", self.answer,
            context=retrieval.passages, question=question,
        )

        tracer.summary()
        # Trace a1b2c3d4:
        #   retrieve: 120ms (15%)
        #   answer: 680ms (85%)
        #   Total: 800ms

        return answer
```

### Save traces for later analysis

```python
def save_trace(tracer, path="traces.jsonl"):
    with open(path, "a") as f:
        f.write(json.dumps(tracer.to_dict()) + "\n")

# Load and analyze traces
def load_traces(path="traces.jsonl"):
    with open(path) as f:
        return [json.loads(line) for line in f]

def find_slow_traces(traces, threshold_ms=2000):
    return [t for t in traces if t["total_latency_ms"] > threshold_ms]

def find_failed_steps(traces):
    return [
        t for t in traces
        if any("error" in str(s.get("outputs", "")).lower() for s in t["steps"])
    ]
```

## Step 4: OpenTelemetry instrumentation

For production tracing with any backend (Jaeger, Zipkin, Datadog, etc.):

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup — do this once at app startup
provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("my-ai-app")

class OTelTracedProgram(dspy.Module):
    """Wraps a DSPy program with OpenTelemetry spans."""
    def __init__(self, program):
        self.program = program

    def forward(self, **kwargs):
        with tracer.start_as_current_span("ai_request") as span:
            span.set_attribute("ai.inputs", json.dumps({k: str(v) for k, v in kwargs.items()}))

            start = time.time()
            result = self.program(**kwargs)
            latency = time.time() - start

            span.set_attribute("ai.latency_ms", round(latency * 1000))
            span.set_attribute("ai.outputs", json.dumps(
                {k: str(getattr(result, k, "")) for k in result.keys()}
            ))

            return result
```

### Trace individual pipeline steps with OTel

```python
class OTelTracedRAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        with tracer.start_as_current_span("rag_pipeline") as parent:
            parent.set_attribute("question", question)

            with tracer.start_as_current_span("retrieve"):
                retrieval = self.retrieve(query=question)

            with tracer.start_as_current_span("generate_answer"):
                answer = self.answer(
                    context=retrieval.passages, question=question
                )

            return answer
```

## Step 5: Connect a trace viewer

### Option A: Langtrace (best DSPy integration)

First-class DSPy auto-instrumentation — one line to trace all LM calls:

```bash
pip install langtrace-python-sdk
```

```python
from langtrace_python_sdk import langtrace

langtrace.init(api_key="your-key")  # or use LANGTRACE_API_KEY env var

# That's it — all DSPy calls are now traced automatically
result = my_program(question="test")
# View traces at app.langtrace.ai
```

### Option B: Arize Phoenix (open-source, self-hosted)

```bash
pip install arize-phoenix openinference-instrumentation-dspy
```

```python
import phoenix as px
from openinference.instrumentation.dspy import DSPyInstrumentor

# Launch local trace viewer
px.launch_app()  # Opens at http://localhost:6006

# Auto-instrument DSPy
DSPyInstrumentor().instrument()

# All DSPy calls are now traced
result = my_program(question="test")
```

### Option C: Jaeger (open-source, Docker)

```bash
docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
```

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Export spans to Jaeger
exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(exporter))

# View traces at http://localhost:16686
```

### Comparison

| Feature | Langtrace | Arize Phoenix | Jaeger |
|---------|-----------|---------------|--------|
| DSPy auto-instrumentation | Yes (built-in) | Yes (plugin) | Manual |
| Setup effort | One line | Two lines + Docker | Docker + manual spans |
| Self-hosted option | Yes | Yes | Yes |
| Cloud option | Yes | Yes | No |
| LM call details | Prompts, tokens, cost | Prompts, tokens | Custom attributes |
| Best for | DSPy-first teams | Teams wanting open-source + UI | Teams already using Jaeger |

## Step 6: Search and filter traces

### Find traces by criteria

```python
def search_traces(traces, **filters):
    """Search traces by user, time range, latency, or content."""
    results = traces

    if "min_latency_ms" in filters:
        results = [t for t in results if t["total_latency_ms"] >= filters["min_latency_ms"]]

    if "after" in filters:
        results = [t for t in results if t["timestamp"] >= filters["after"]]

    if "before" in filters:
        results = [t for t in results if t["timestamp"] <= filters["before"]]

    if "contains" in filters:
        keyword = filters["contains"].lower()
        results = [
            t for t in results
            if keyword in json.dumps(t).lower()
        ]

    return results

# Find slow requests from today
slow = search_traces(
    load_traces(),
    min_latency_ms=3000,
    after="2025-01-15T00:00:00",
)
```

### Aggregate trace statistics

```python
def trace_stats(traces):
    """Summary statistics across traces."""
    latencies = [t["total_latency_ms"] for t in traces]
    if not latencies:
        return "No traces found"

    latencies.sort()
    return {
        "count": len(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "max_ms": latencies[-1],
    }
```

## Step 7: Use traces to improve your AI

Traces aren't just for debugging — they're a source of improvement.

### Find patterns in wrong answers

```python
# Load traces where the answer was marked wrong by a user or metric
wrong_traces = search_traces(load_traces(), contains='"is_correct": false')

# Check which step is most often the bottleneck
from collections import Counter
slow_steps = Counter()
for t in wrong_traces:
    slowest = max(t["steps"], key=lambda s: s["latency_ms"])
    slow_steps[slowest["step"]] += 1

print(slow_steps)
# Counter({"retrieve": 23, "answer": 7})
# -> Retrieval is the problem, not the answer generation
```

### Build training data from failures

```python
# Extract failed examples for re-optimization
failed_examples = []
for t in wrong_traces:
    ex = dspy.Example(
        question=t.get("inputs", {}).get("question", ""),
    ).with_inputs("question")
    failed_examples.append(ex)

# Add to training set and re-optimize
# See /ai-improving-accuracy
```

## Key patterns

- **Start with `dspy.inspect_history`** — it's free and solves most debugging needs
- **Add JSONL tracing before you need it** — you can't debug traces you didn't log
- **Trace at the step level**, not just the request level — per-step latency reveals bottlenecks
- **Use OpenTelemetry for production** — it's the standard, works with any backend
- **Langtrace is easiest for DSPy** — one-line setup with automatic instrumentation
- **Traces feed optimization** — patterns in wrong answers tell you what to fix

## Additional resources

- For worked examples, see [examples.md](examples.md)
- Use `/ai-monitoring` for aggregate health checks across all requests
- Use `/ai-fixing-errors` for code-level debugging (crashes, config issues)
- Use `/ai-building-pipelines` to structure pipelines that are easy to trace
- Use `/ai-improving-accuracy` to optimize based on patterns found in traces
