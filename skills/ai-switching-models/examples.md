# Examples: Switching Models

## Example 1: Cost migration — GPT-4o to GPT-4o-mini

A support ticket classifier running on GPT-4o costs too much. Let's switch to GPT-4o-mini and see if quality holds.

### Setup

```python
import dspy
from dspy.evaluate import Evaluate

# The task: classify support tickets
class ClassifyTicket(dspy.Signature):
    """Classify a support ticket into a category."""
    ticket_text: str = dspy.InputField()
    category: str = dspy.OutputField(desc="one of: billing, technical, account, feature_request, other")

class TicketClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(ClassifyTicket)

    def forward(self, ticket_text):
        return self.classify(ticket_text=ticket_text)

# Metric
def metric(example, prediction, trace=None):
    return prediction.category.strip().lower() == example.category.strip().lower()

# Test data (50+ examples for reliable evaluation)
devset = [
    dspy.Example(ticket_text="I was charged twice for my subscription", category="billing").with_inputs("ticket_text"),
    dspy.Example(ticket_text="The API returns 500 errors", category="technical").with_inputs("ticket_text"),
    # ... more examples
]
trainset = devset[:40]  # for optimization
testset = devset[40:]   # held out for final eval

evaluator = Evaluate(devset=testset, metric=metric, num_threads=4, display_progress=True)
```

### Step 1: Benchmark GPT-4o (the expensive model)

```python
gpt4o = dspy.LM("openai/gpt-4o")
dspy.configure(lm=gpt4o)

# Optimize for GPT-4o
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized_gpt4o = optimizer.compile(TicketClassifier(), trainset=trainset)

baseline = evaluator(optimized_gpt4o)
print(f"GPT-4o (optimized): {baseline:.1f}%")
# GPT-4o (optimized): 92.0%
```

### Step 2: Try GPT-4o-mini with GPT-4o's prompts (see the drop)

```python
gpt4o_mini = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=gpt4o_mini)

# Use GPT-4o's optimized prompts on GPT-4o-mini — no re-optimization
naive_score = evaluator(optimized_gpt4o)
print(f"GPT-4o-mini (GPT-4o's prompts): {naive_score:.1f}%")
# GPT-4o-mini (GPT-4o's prompts): 78.0%  <-- quality dropped!
```

The optimized prompts from GPT-4o don't work well on GPT-4o-mini. This is the core finding from the research — prompts are model-specific.

### Step 3: Re-optimize for GPT-4o-mini

```python
dspy.configure(lm=gpt4o_mini)

# Fresh program, optimize specifically for GPT-4o-mini
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized_mini = optimizer.compile(TicketClassifier(), trainset=trainset)

reoptimized_score = evaluator(optimized_mini)
print(f"\n--- Results ---")
print(f"GPT-4o (optimized):              {baseline:.1f}%")
print(f"GPT-4o-mini (old prompts):       {naive_score:.1f}%")
print(f"GPT-4o-mini (re-optimized):      {reoptimized_score:.1f}%")
# GPT-4o (optimized):              92.0%
# GPT-4o-mini (old prompts):       78.0%   <-- 14% drop without re-optimization
# GPT-4o-mini (re-optimized):      89.0%   <-- recovered most of the quality
```

### Step 4: Ship it

```python
# 89% vs 92% — close enough for a 33x cost reduction
optimized_mini.save("ticket_classifier_gpt4o_mini.json")

# In production
classifier = TicketClassifier()
classifier.load("ticket_classifier_gpt4o_mini.json")
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
result = classifier(ticket_text="I can't log in to my account")
```

**Key takeaway:** Without re-optimization, switching to GPT-4o-mini dropped quality by 14%. With re-optimization, the drop was only 3% — an acceptable trade-off for 33x cost savings.

---

## Example 2: Vendor switch — OpenAI to Anthropic

The team wants to reduce dependency on OpenAI. Let's migrate a Q&A system to Anthropic.

### Setup

```python
import dspy
from dspy.evaluate import Evaluate

class AnswerQuestion(dspy.Signature):
    """Answer the question based on the given context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class QASystem(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought(AnswerQuestion)

    def forward(self, context, question):
        return self.answer(context=context, question=question)

# F1 metric for text overlap
def metric(example, prediction, trace=None):
    gold_tokens = set(example.answer.lower().split())
    pred_tokens = set(prediction.answer.lower().split())
    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)
    precision = len(gold_tokens & pred_tokens) / len(pred_tokens)
    recall = len(gold_tokens & pred_tokens) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

evaluator = Evaluate(devset=testset, metric=metric, num_threads=4, display_progress=True)
```

### Step 1: Benchmark OpenAI baseline

```python
openai_lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=openai_lm)

optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized_openai = optimizer.compile(QASystem(), trainset=trainset)

openai_score = evaluator(optimized_openai)
print(f"OpenAI GPT-4o (optimized): {openai_score:.1f}%")
# OpenAI GPT-4o (optimized): 85.2%
```

### Step 2: Try Anthropic with OpenAI's prompts

```python
claude_lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929")
dspy.configure(lm=claude_lm)

# OpenAI's optimized prompts on Claude
naive_score = evaluator(optimized_openai)
print(f"Claude (OpenAI's prompts): {naive_score:.1f}%")
# Claude (OpenAI's prompts): 76.8%
```

### Step 3: Re-optimize for Anthropic

```python
dspy.configure(lm=claude_lm)

optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized_claude = optimizer.compile(QASystem(), trainset=trainset)

claude_score = evaluator(optimized_claude)
print(f"\n--- Results ---")
print(f"OpenAI GPT-4o (optimized):     {openai_score:.1f}%")
print(f"Claude (OpenAI's prompts):      {naive_score:.1f}%")
print(f"Claude (re-optimized):          {claude_score:.1f}%")
# OpenAI GPT-4o (optimized):     85.2%
# Claude (OpenAI's prompts):      76.8%  <-- 8.4% drop
# Claude (re-optimized):          86.1%  <-- actually beat the original!
```

### Step 4: Deploy with model selection

```python
# Save both optimized programs
optimized_openai.save("qa_openai_gpt4o.json")
optimized_claude.save("qa_anthropic_claude.json")

# In production — select model via environment variable
import os

model_configs = {
    "openai": ("openai/gpt-4o", "qa_openai_gpt4o.json"),
    "anthropic": ("anthropic/claude-sonnet-4-5-20250929", "qa_anthropic_claude.json"),
}

provider = os.environ.get("AI_PROVIDER", "anthropic")
model_id, program_path = model_configs[provider]

lm = dspy.LM(model_id)
dspy.configure(lm=lm)

qa = QASystem()
qa.load(program_path)
```

**Key takeaway:** Swapping providers without re-optimization lost 8.4%. After re-optimization, Claude actually scored higher. The DSPy program (signatures + modules) didn't change at all — only the compiled prompts did.

---

## Example 3: Model shootout — compare 4 models

Choosing a model for a new feature? Run a systematic comparison.

### Setup

```python
import dspy
from dspy.evaluate import Evaluate

class Summarize(dspy.Signature):
    """Summarize the article in 2-3 sentences."""
    article: str = dspy.InputField()
    summary: str = dspy.OutputField()

class Summarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(Summarize)

    def forward(self, article):
        return self.summarize(article=article)

# AI-as-judge metric
class AssessSummary(dspy.Signature):
    """Assess if the summary captures the key points of the article."""
    article: str = dspy.InputField()
    gold_summary: str = dspy.InputField()
    predicted_summary: str = dspy.InputField()
    is_good: bool = dspy.OutputField()

# Use a strong model as the judge (separate from candidates)
judge_lm = dspy.LM("openai/gpt-4o")

def metric(example, prediction, trace=None):
    with dspy.context(lm=judge_lm):
        judge = dspy.Predict(AssessSummary)
        result = judge(
            article=example.article,
            gold_summary=example.summary,
            predicted_summary=prediction.summary,
        )
    return result.is_good

evaluator = Evaluate(devset=testset, metric=metric, num_threads=4, display_progress=True)
```

### Run the shootout

```python
candidates = [
    ("openai/gpt-4o", "GPT-4o"),
    ("openai/gpt-4o-mini", "GPT-4o-mini"),
    ("anthropic/claude-sonnet-4-5-20250929", "Claude Sonnet"),
    ("together_ai/meta-llama/Llama-3-70b-chat-hf", "Llama 3 70B"),
]

results = []

for model_id, label in candidates:
    print(f"\n{'='*40}")
    print(f"Testing: {label}")
    print(f"{'='*40}")

    lm = dspy.LM(model_id)
    dspy.configure(lm=lm)

    # Quick optimization (BootstrapFewShot for speed)
    fresh = Summarizer()
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
    )
    optimized = optimizer.compile(fresh, trainset=trainset)

    # Evaluate
    score = evaluator(optimized)

    # Save
    optimized.save(f"summarizer_{label.lower().replace(' ', '_')}.json")

    results.append({
        "model": label,
        "model_id": model_id,
        "score": score,
    })
```

### Compare results

```python
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f"{'Model':<25} {'Score':>8}")
print("-" * 35)
for r in sorted(results, key=lambda x: x["score"], reverse=True):
    print(f"{r['model']:<25} {r['score']:>7.1f}%")

# Example output:
# MODEL COMPARISON
# ==================================================
# Model                      Score
# -----------------------------------
# Claude Sonnet               88.0%
# GPT-4o                      86.0%
# Llama 3 70B                 82.0%
# GPT-4o-mini                 79.0%
```

### Deeper comparison with MIPROv2

If BootstrapFewShot results are close, run MIPROv2 on the top contenders for a more accurate comparison:

```python
top_models = [r for r in sorted(results, key=lambda x: x["score"], reverse=True)[:2]]

for r in top_models:
    lm = dspy.LM(r["model_id"])
    dspy.configure(lm=lm)

    fresh = Summarizer()
    optimizer = dspy.MIPROv2(metric=metric, auto="medium")
    optimized = optimizer.compile(fresh, trainset=trainset)

    score = evaluator(optimized)
    optimized.save(f"summarizer_{r['model'].lower().replace(' ', '_')}_mipro.json")
    print(f"{r['model']} (MIPROv2): {score:.1f}%")
```

**Key takeaway:** Always optimize per-model before comparing. Comparing models with unoptimized (or another model's) prompts gives misleading results. The ranking can change after optimization.
