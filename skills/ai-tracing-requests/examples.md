# Tracing Examples

## Example 1: Debugging a RAG pipeline wrong answer

A customer reported that the help center bot gave a wrong answer about refund policies. Walk through tracing the exact request.

### The problem

```python
result = help_bot(question="Can I get a refund after 30 days?")
print(result.answer)
# "Yes, you can request a refund at any time."
# WRONG — the actual policy is 30-day limit
```

### Step 1: Inspect the LM calls

```python
# Re-run the question and inspect
result = help_bot(question="Can I get a refund after 30 days?")
dspy.inspect_history(n=3)
```

Output shows:
```
--- LM Call 1 (retrieve) ---
Query: "Can I get a refund after 30 days?"
Retrieved passages:
  1. "Refund requests must be submitted within 30 days..."  ✓ correct
  2. "We offer a satisfaction guarantee on all products..."  ✗ irrelevant
  3. "Contact support@example.com for assistance..."        ✗ irrelevant

--- LM Call 2 (answer) ---
Prompt: "Answer the question based on the context..."
Context: [the 3 passages above]
Response: "Yes, you can request a refund at any time."
```

**Root cause found:** The retriever found the right document (passage 1), but the other 2 passages diluted the context. The LM ignored the 30-day limit mentioned in passage 1.

### Step 2: Trace with timing

```python
class TracedHelpBot(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        tracer = StepTracer()

        retrieval = tracer.trace_step("retrieve", self.retrieve, query=question)
        print(f"Retrieved {len(retrieval.passages)} passages:")
        for i, p in enumerate(retrieval.passages):
            print(f"  [{i+1}] {p[:100]}...")

        answer = tracer.trace_step(
            "answer", self.answer,
            context=retrieval.passages, question=question,
        )

        tracer.summary()
        save_trace(tracer)
        return answer

bot = TracedHelpBot()
result = bot(question="Can I get a refund after 30 days?")
# Retrieved 3 passages:
#   [1] Refund requests must be submitted within 30 days of purchase...
#   [2] We offer a satisfaction guarantee on all products...
#   [3] Contact support@example.com for assistance...
# Trace a1b2c3d4:
#   retrieve: 95ms (12%)
#   answer: 720ms (88%)
#   Total: 815ms
```

### Step 3: Fix the issue

The fix: reduce k to 2 to get more focused context, and add a constraint:

```python
class FixedHelpBot(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=2)  # fewer, more relevant passages
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        retrieval = self.retrieve(query=question)
        answer = self.answer(context=retrieval.passages, question=question)

        # Ensure the answer references the actual policy
        dspy.Suggest(
            any(p[:50] in answer.answer for p in retrieval.passages)
            or len(answer.answer) > 20,
            "Answer should reference specific information from the documents"
        )

        return answer
```

## Example 2: Profiling a slow multi-step pipeline

A classification pipeline takes 8+ seconds. Find the bottleneck.

### The slow pipeline

```python
class ContentPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought("text -> category")
        self.extract = dspy.ChainOfThought("text, category -> entities: list[str]")
        self.summarize = dspy.ChainOfThought("text, category, entities -> summary")
        self.check = dspy.ChainOfThought("text, summary -> is_safe: bool, issues: list[str]")

    def forward(self, text):
        cat = self.classify(text=text)
        ents = self.extract(text=text, category=cat.category)
        summary = self.summarize(text=text, category=cat.category, entities=ents.entities)
        safety = self.check(text=text, summary=summary.summary)
        return dspy.Prediction(
            category=cat.category,
            entities=ents.entities,
            summary=summary.summary,
            is_safe=safety.is_safe,
        )
```

### Add tracing to find the bottleneck

```python
class ProfiledPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought("text -> category")
        self.extract = dspy.ChainOfThought("text, category -> entities: list[str]")
        self.summarize = dspy.ChainOfThought("text, category, entities -> summary")
        self.check = dspy.ChainOfThought("text, summary -> is_safe: bool, issues: list[str]")

    def forward(self, text):
        tracer = StepTracer()

        cat = tracer.trace_step("classify", self.classify, text=text)
        ents = tracer.trace_step("extract", self.extract, text=text, category=cat.category)
        summary = tracer.trace_step(
            "summarize", self.summarize,
            text=text, category=cat.category, entities=ents.entities,
        )
        safety = tracer.trace_step("safety_check", self.check, text=text, summary=summary.summary)

        tracer.summary()
        save_trace(tracer)

        return dspy.Prediction(
            category=cat.category,
            entities=ents.entities,
            summary=summary.summary,
            is_safe=safety.is_safe,
        )

pipeline = ProfiledPipeline()
result = pipeline(text="Long article text here...")

# Trace output:
# Trace f3e4d5c6:
#   classify: 450ms (5%)
#   extract: 1200ms (14%)
#   summarize: 5800ms (68%)    <-- BOTTLENECK
#   safety_check: 1100ms (13%)
#   Total: 8550ms
```

**Bottleneck found:** The summarize step takes 68% of total time.

### Fix: use a cheaper model for the bottleneck

```python
expensive_lm = dspy.LM("openai/gpt-4o")
cheap_lm = dspy.LM("openai/gpt-4o-mini")

pipeline = ProfiledPipeline()

# Use cheap model for the slow step (summarization is easier than classification)
pipeline.summarize.set_lm(cheap_lm)

# Re-profile
result = pipeline(text="Long article text here...")
# Trace g7h8i9j0:
#   classify: 450ms (10%)
#   extract: 1200ms (28%)
#   summarize: 1400ms (32%)   <-- 4x faster!
#   safety_check: 1100ms (30%)
#   Total: 4150ms             <-- 51% reduction
```

### Run profiling across multiple inputs

```python
traces = []
for text in test_texts[:20]:
    pipeline(text=text)

all_traces = load_traces()
stats = trace_stats(all_traces)
print(stats)
# {"count": 20, "p50_ms": 4200, "p95_ms": 6800, "p99_ms": 8100, "max_ms": 8550}

# Find which step is slowest across all traces
from collections import defaultdict
step_times = defaultdict(list)
for t in all_traces:
    for step in t["steps"]:
        step_times[step["step"]].append(step["latency_ms"])

for step, times in step_times.items():
    times.sort()
    p50 = times[len(times) // 2]
    print(f"  {step}: p50={p50}ms")
```
