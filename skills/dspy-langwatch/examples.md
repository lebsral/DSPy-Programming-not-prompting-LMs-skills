# LangWatch Examples

## Trace a pipeline at inference

```python
import langwatch
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

class SupportBot(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)

bot = SupportBot()

@langwatch.trace()
def handle_support(question):
    langwatch.get_current_trace().autotrack_dspy()
    return bot(question=question)

# Run queries — all automatically traced
questions = [
    "How do I reset my password?",
    "What's your refund policy?",
    "Can I upgrade my plan mid-cycle?",
]

for q in questions:
    result = handle_support(question=q)
    print(f"Q: {q}\nA: {result.answer}\n")
```

### Find slow requests in the LangWatch UI

1. Go to [app.langwatch.ai](https://app.langwatch.ai) (or your self-hosted URL)
2. Open your project
3. Sort traces by latency (descending)
4. Click a slow trace to see the span tree
5. Check which step is the bottleneck:
   - **Retrieve slow?** Vector DB may need optimization or query is too broad
   - **LM call slow?** Model may be overloaded or prompt is too long

## Watch MIPROv2 optimization live

```python
import langwatch.dspy
import dspy
from dspy.evaluate import Evaluate

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

# Prepare data
trainset = [
    dspy.Example(question="What is Python?", answer="A programming language").with_inputs("question"),
    dspy.Example(question="What is DSPy?", answer="A framework for programming LMs").with_inputs("question"),
    # ... more examples
]
devset = trainset[:10]

def metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# Set up optimizer with LangWatch tracking
program = dspy.ChainOfThought("question -> answer")
optimizer = dspy.MIPROv2(metric=metric, auto="medium")

langwatch.dspy.init(
    experiment="mipro-medium-support-bot",
    optimizer=optimizer,
)

# Start optimization — open app.langwatch.ai to watch live
optimized = optimizer.compile(program, trainset=trainset)

# The dashboard shows:
# - Each trial's score as it completes
# - Which instructions/demos the optimizer tested
# - Running cost total
# - Progress through the optimization

# Evaluate the result
evaluator = Evaluate(devset=devset, metric=metric, num_threads=4)
score = evaluator(optimized)
print(f"Final score: {score:.1f}%")
```

## Compare optimizer strategies side-by-side

```python
import langwatch.dspy
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

def metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# Run three experiments — each appears in the LangWatch dashboard
strategies = [
    {
        "name": "bootstrap-quick",
        "optimizer_class": dspy.BootstrapFewShot,
        "kwargs": {"metric": metric, "max_bootstrapped_demos": 4},
    },
    {
        "name": "mipro-light",
        "optimizer_class": dspy.MIPROv2,
        "kwargs": {"metric": metric, "auto": "light"},
    },
    {
        "name": "mipro-medium",
        "optimizer_class": dspy.MIPROv2,
        "kwargs": {"metric": metric, "auto": "medium"},
    },
]

results = {}
for strategy in strategies:
    program = dspy.ChainOfThought("question -> answer")
    optimizer = strategy["optimizer_class"](**strategy["kwargs"])

    langwatch.dspy.init(
        experiment=strategy["name"],
        optimizer=optimizer,
    )

    optimized = optimizer.compile(program, trainset=trainset)
    results[strategy["name"]] = optimized

# Open app.langwatch.ai — all three experiments are visible
# Compare scores, cost, and convergence speed across strategies
```

## Trace with metadata in production

```python
import langwatch
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

bot = SupportBot()

@langwatch.trace()
def handle_support(user_id, plan, question):
    trace = langwatch.get_current_trace()
    trace.autotrack_dspy()
    trace.update(metadata={
        "user_id": user_id,
        "plan": plan,
        "source": "api",
    })
    return bot(question=question)

# In production — filter traces by plan or user_id in the dashboard
result = handle_support("user-456", "enterprise", "How do I set up SSO?")
```
