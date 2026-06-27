# Fixing Broken DSPy Programs — Examples

## Could Not Parse Output

Symptom: `ValueError: Could not parse output` when calling a DSPy module.

First, inspect what the LM actually returned before touching any code:

```python
dspy.inspect_history(n=1)
```

The raw response is usually in the wrong format — not a model problem, a signature problem.

**Fix A — add descriptive field descriptions:**

```python
# Before — vague field, LM guesses format
class ExtractItems(dspy.Signature):
    text: str = dspy.InputField()
    items: list[str] = dspy.OutputField()

# After — LM knows exactly what to produce
class ExtractItems(dspy.Signature):
    """Extract the action items from a meeting transcript."""
    text: str = dspy.InputField(desc="meeting transcript text")
    items: list[str] = dspy.OutputField(desc="each action item as a short, separate string")
```

**Fix B — switch from Predict to ChainOfThought:**

`dspy.Predict` asks the LM to produce structured output immediately. `dspy.ChainOfThought` lets it reason first, which significantly improves format compliance for structured types like `list[str]` or Pydantic models.

```python
# Before
extractor = dspy.Predict(ExtractItems)

# After
extractor = dspy.ChainOfThought(ExtractItems)
```

---

## Context Window Exceeded

Symptom: `ContextLengthExceededError` (or similar), usually in RAG pipelines or optimized programs with many few-shot demos.

Two levers — reduce retrieved context, reduce demos:

```python
# Retrieve fewer passages
self.retrieve = dspy.Retrieve(k=2)  # was k=10

# Or re-optimize with fewer bootstrapped demos
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
compiled = optimizer.compile(program, trainset=trainset)
```

If the prompt is still too long after reducing `k`, truncate the input before passing it in, or switch to a model with a larger context window.

---

## RateLimitError and Intermittent Failures

Symptom: `RateLimitError`, `TimeoutError`, or the program succeeds sometimes and fails other times.

DSPy caches LM calls by default — repeated identical calls don't hit the API. Intermittent failures usually mean parallel threads are hitting rate limits on the first pass.

```python
# Reduce parallel threads during evaluation or optimization
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=devset, metric=metric, num_threads=2, display_progress=True)

# Add a timeout so hung calls fail fast instead of blocking forever
lm = dspy.LM("openai/gpt-4o-mini", timeout=30)  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

For optimization runs that hit limits, reduce trainset size during iteration and use the full dataset for the final compile.

---

## TypeError - Unexpected Keyword Argument

Symptom: `TypeError: forward() got an unexpected keyword argument 'query'`

The keyword you passed doesn't match any `InputField` name in the signature.

```python
class SearchDocs(dspy.Signature):
    """Find the answer in the documentation."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

searcher = dspy.Predict(SearchDocs)

# Wrong — 'query' is not an InputField
result = searcher(query="how do I configure DSPy?")

# Right — must match the InputField name exactly
result = searcher(question="how do I configure DSPy?")
```

Print `MySignature.fields` to see the exact names if you're not sure.

---

## End-to-End Debugging Walkthrough

A two-step pipeline (query generation then answer generation) that returns garbage on the answer step. This walkthrough uses `dspy.inspect_history` to find the root cause without guessing.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or any supported provider
dspy.configure(lm=lm)

class GenerateQuery(dspy.Signature):
    """Turn a user question into a concise search query."""
    question: str = dspy.InputField()
    query: str = dspy.OutputField()

class GenerateAnswer(dspy.Signature):
    """Answer the question using only the retrieved passages."""
    context: str = dspy.InputField(desc="retrieved passages, one per line")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class RAGPipeline(dspy.Module):
    def __init__(self):
        self.to_query = dspy.ChainOfThought(GenerateQuery)
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        step1 = self.to_query(question=question)
        print(f"[step1] query: {step1.query}")  # isolate step 1

        docs = self.retrieve(step1.query)
        print(f"[step2] retrieved: {len(docs.passages)} passages")  # isolate step 2

        context = "\n".join(docs.passages)
        return self.answer(context=context, question=question)

pipeline = RAGPipeline()
result = pipeline(question="What optimizer should I use for a small dataset?")
```

After running, inspect the last few LM calls:

```python
dspy.inspect_history(n=3)
```

What to look for in the raw output:

| What you see in inspect_history | Diagnosis | Fix |
|---|---|---|
| Context field is empty or `"[]"` | Retriever returned no passages | Call the retriever directly to verify: `rm("test query", k=3)` |
| Raw response is prose, not a field value | Parse error — format mismatch | Add `desc=` to the output field, or switch to `ChainOfThought` |
| Response truncated mid-sentence | Context window exceeded | Reduce `k`, truncate `context` before passing |
| Correct response but field access raises `AttributeError` | Wrong field name on result | Print `result` to see what fields were actually returned |

Once the pipeline runs cleanly, verify the module structure and check that every predictor was tracked:

```python
for name, predictor in pipeline.named_predictors():
    print(f"{name}: {type(predictor).__name__}")
# to_query: ChainOfThought
# answer: ChainOfThought

print(result.answer)
```

If a predictor is missing from the list, it was constructed inside `forward()` instead of `__init__()` — move it to `__init__` so DSPy can track and optimize it.
