# Weave Examples

## Track optimization experiments and compare runs

### Setup

```python
import weave
import dspy
from dspy.evaluate import Evaluate

weave.init("dspy-experiments")
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Shared data and metric
trainset = [
    dspy.Example(question="What is Python?", answer="A programming language.").with_inputs("question"),
    # ... 50+ examples
]
devset = trainset[:20]

def metric(example, prediction, trace=None):
    judge = dspy.Predict("gold_answer, predicted_answer -> match: bool")
    result = judge(gold_answer=example.answer, predicted_answer=prediction.answer)
    return result.match
```

### Run multiple experiments

```python
@weave.op()
def experiment(name: str, optimizer_type: str, auto: str = "light"):
    """Each call creates a tracked run in Weave."""
    program = dspy.ChainOfThought("question -> answer")

    if optimizer_type == "miprov2":
        optimizer = dspy.MIPROv2(metric=metric, auto=auto)
        optimized = optimizer.compile(program, trainset=trainset)
    elif optimizer_type == "bootstrap":
        optimizer = dspy.BootstrapFewShot(metric=metric)
        optimized = optimizer.compile(program, trainset=trainset)
    elif optimizer_type == "gepa":
        def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            score = float(metric(gold, pred))
            return {"score": score, "feedback": "" if score else "Wrong answer."}
        optimizer = dspy.GEPA(metric=gepa_metric, auto=auto)
        optimized = optimizer.compile(program, trainset=trainset)

    evaluator = Evaluate(devset=devset, metric=metric, num_threads=4)
    score = evaluator(optimized)

    optimized.save(f"experiments/{name}.json")
    return {"name": name, "score": score, "optimizer": optimizer_type, "auto": auto}

# Run experiments
experiment("baseline-bootstrap", "bootstrap")
experiment("mipro-light", "miprov2", "light")
experiment("mipro-medium", "miprov2", "medium")
experiment("gepa-light", "gepa", "light")
```

### Compare in W&B dashboard

1. Go to [wandb.ai](https://wandb.ai) → your project → "Traces" tab
2. You'll see 4 tracked function calls with their inputs and outputs
3. Click each to see:
   - Input parameters (optimizer type, auto setting)
   - Output (score, artifact path)
   - Latency (how long optimization took)
   - Cost (token usage)
4. Sort by output score to find the winner
5. Click "Share" to send the dashboard URL to your team

## Track production queries with metadata

```python
import weave
import dspy

weave.init("production-qa")
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

program = dspy.ChainOfThought("question -> answer")
program.load("experiments/mipro-medium.json")

@weave.op()
def handle_query(user_id: str, question: str, source: str = "api"):
    """Production query handler — every call tracked in Weave."""
    result = program(question=question)
    return {
        "answer": result.answer,
        "user_id": user_id,
        "source": source,
    }

# Production usage
handle_query("user-42", "What's the return policy?", source="web")
handle_query("user-99", "How do I upgrade?", source="mobile")

# In the Weave dashboard:
# - Filter by source to compare web vs mobile usage
# - Sort by latency to find slow queries
# - Click into a trace to see the full LM prompt/response
```
