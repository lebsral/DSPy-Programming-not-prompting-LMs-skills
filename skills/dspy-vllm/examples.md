# vLLM Examples

## Start a vLLM server and connect DSPy

```bash
# Install
pip install vllm

# Serve Llama 3.1 8B
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching
```

```python
import dspy

lm = dspy.LM(
    "openai/meta-llama/Llama-3.1-8B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="none",
    temperature=0.7,
    max_tokens=1000,
)
dspy.configure(lm=lm)

# Build a simple pipeline
class TicketRouter(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("ticket -> category, priority")
        self.respond = dspy.ChainOfThought("ticket, category, priority -> response")

    def forward(self, ticket):
        triage = self.classify(ticket=ticket)
        return self.respond(
            ticket=ticket,
            category=triage.category,
            priority=triage.priority,
        )

router = TicketRouter()
result = router(ticket="I was charged twice and need a refund immediately")
print(f"Category: {result.category}")
print(f"Priority: {result.priority}")
print(f"Response: {result.response}")
```

## Multi-GPU serving (70B model)

```bash
# Serve Llama 3.1 70B on 2x A100-80GB
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching

# Or quantized on a single A100-80GB
vllm serve TheBloke/Llama-2-70B-Chat-AWQ \
    --quantization awq \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

```python
import dspy

# Connect to the 70B model — same DSPy API
lm = dspy.LM(
    "openai/meta-llama/Llama-3.1-70B-Instruct",
    api_base="http://gpu-server:8000/v1",
    api_key="none",
    max_tokens=2000,
)
dspy.configure(lm=lm)

# Complex reasoning benefits from larger models
analyze = dspy.ChainOfThought("document -> summary, key_findings: list[str], risk_level")
result = analyze(document="... long contract text ...")
print(result.summary)
print(result.key_findings)
print(result.risk_level)
```

## Optimize with vLLM backend

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM(
    "openai/meta-llama/Llama-3.1-8B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="none",
    max_tokens=500,
)
dspy.configure(lm=lm)

# Prepare data
trainset = [
    dspy.Example(question="What is Python?", answer="A programming language").with_inputs("question"),
    dspy.Example(question="What is DSPy?", answer="A framework for programming LMs").with_inputs("question"),
    dspy.Example(question="What is vLLM?", answer="A high-throughput LLM serving engine").with_inputs("question"),
    # ... add 50+ examples for stable optimization
]
devset = trainset[:10]

def metric(example, prediction, trace=None):
    from dspy.evaluate import SemanticF1
    return SemanticF1()(example, prediction)

# Baseline
program = dspy.ChainOfThought("question -> answer")
evaluator = Evaluate(devset=devset, metric=metric, num_threads=8)
baseline = evaluator(program)
print(f"Baseline: {baseline:.1f}%")

# Optimize — vLLM handles concurrent optimizer calls efficiently
optimizer = dspy.MIPROv2(metric=metric, auto="light")
optimized = optimizer.compile(program, trainset=trainset)

optimized_score = evaluator(optimized)
print(f"Optimized: {optimized_score:.1f}%")
print(f"Delta: {optimized_score - baseline:+.1f}%")

# Save
optimized.save("optimized_qa.json")
```
