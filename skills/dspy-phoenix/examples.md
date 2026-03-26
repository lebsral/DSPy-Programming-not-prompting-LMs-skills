# Phoenix Examples

## Trace a DSPy pipeline and inspect in the UI

### Setup

```python
import phoenix as px
from openinference.instrumentation.dspy import DSPyInstrumentor

px.launch_app()  # http://localhost:6006
DSPyInstrumentor().instrument()

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

### Build and run a pipeline

```python
class QAPipeline(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)

pipeline = QAPipeline()

# Run several queries to populate traces
questions = [
    "What is our return policy?",
    "How do I cancel my subscription?",
    "What payment methods do you accept?",
    "How long does shipping take?",
]

for q in questions:
    result = pipeline(question=q)
    print(f"Q: {q}\nA: {result.answer}\n")
```

### Inspect traces in the Phoenix UI

1. Open `http://localhost:6006` in your browser
2. You'll see 4 traces in the trace list
3. Click any trace to see the span tree:
   - **Root span**: `QAPipeline.forward()`
   - **Child span 1**: `Retrieve` — shows the query and retrieved passages
   - **Child span 2**: `ChainOfThought` — shows the full prompt, response, token count
4. Sort by latency to find the slowest request
5. Check token counts to identify expensive queries

## Evaluate trace quality with Phoenix evals

```python
import phoenix as px
from phoenix.evals import llm_classify, OpenAIModel

# Get traces as a DataFrame
client = px.Client()
spans_df = client.get_spans_dataframe()

# Filter to LM output spans
lm_spans = spans_df[spans_df["span_kind"] == "LLM"]

# Evaluate helpfulness
eval_model = OpenAIModel(model="gpt-4o-mini")
results = llm_classify(
    dataframe=lm_spans,
    model=eval_model,
    template=(
        "Given this question and answer, is the answer helpful and complete?\n"
        "Question: {attributes.input.value}\n"
        "Answer: {attributes.output.value}\n"
    ),
    rails=["helpful", "not helpful"],
)

# Check results
print(results.value_counts())
# helpful        3
# not helpful    1

# Find the unhelpful response
unhelpful = results[results["label"] == "not helpful"]
print(f"Unhelpful trace IDs: {unhelpful.index.tolist()}")
```

## Compare traces before and after optimization

```python
import phoenix as px
from openinference.instrumentation.dspy import DSPyInstrumentor

px.launch_app()
DSPyInstrumentor().instrument()

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

pipeline = QAPipeline()

# Run baseline
print("--- Baseline ---")
for q in questions:
    result = pipeline(question=q)

# Optimize
optimizer = dspy.BootstrapFewShot(metric=my_metric)
optimized = optimizer.compile(pipeline, trainset=trainset)

# Run optimized
print("--- Optimized ---")
for q in questions:
    result = optimized(question=q)

# In Phoenix UI:
# - Filter by time to compare before/after traces
# - Check if token counts changed (optimized may use more tokens for demos)
# - Compare latency distributions
```
