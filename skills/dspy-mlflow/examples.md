# MLflow Examples

## Autolog a DSPy pipeline and view traces

### Setup

```python
import mlflow

mlflow.dspy.autolog()

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class SupportQA(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)

qa = SupportQA()
```

### Run queries (all automatically traced)

```python
questions = [
    "How do I reset my password?",
    "What's your refund policy?",
    "Can I upgrade mid-cycle?",
]

for q in questions:
    result = qa(question=q)
    print(f"Q: {q}\nA: {result.answer}\n")
```

### View traces

```bash
mlflow ui  # Open http://localhost:5000
```

Click the "Traces" tab to see:
- Each query as a separate trace
- Expand a trace to see Retrieve and ChainOfThought spans
- Click an LM span to see the full prompt and response
- Check token counts and latency per span

## Track optimization runs and pick the winner

```python
import mlflow
import dspy
from dspy.evaluate import Evaluate

mlflow.dspy.autolog()
mlflow.set_experiment("qa-optimization")

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

trainset = [
    dspy.Example(question="How do refunds work?", answer="Full refund within 30 days.").with_inputs("question"),
    # ... 50+ examples
]
devset = trainset[:20]

def metric(example, prediction, trace=None):
    judge = dspy.Predict("gold, predicted -> match: bool")
    return judge(gold=example.answer, predicted=prediction.answer).match

# Experiment 1: BootstrapFewShot
with mlflow.start_run(run_name="bootstrap"):
    mlflow.log_param("optimizer", "BootstrapFewShot")

    program = dspy.ChainOfThought("question -> answer")
    optimizer = dspy.BootstrapFewShot(metric=metric)
    optimized = optimizer.compile(program, trainset=trainset)

    score = Evaluate(devset=devset, metric=metric, num_threads=4)(optimized)
    mlflow.log_metric("dev_score", score)

    optimized.save("bootstrap_optimized.json")
    mlflow.log_artifact("bootstrap_optimized.json")

# Experiment 2: MIPROv2 light
with mlflow.start_run(run_name="mipro-light"):
    mlflow.log_param("optimizer", "MIPROv2")
    mlflow.log_param("auto", "light")

    program = dspy.ChainOfThought("question -> answer")
    optimizer = dspy.MIPROv2(metric=metric, auto="light")
    optimized = optimizer.compile(program, trainset=trainset)

    score = Evaluate(devset=devset, metric=metric, num_threads=4)(optimized)
    mlflow.log_metric("dev_score", score)

    optimized.save("mipro_light_optimized.json")
    mlflow.log_artifact("mipro_light_optimized.json")

# Experiment 3: MIPROv2 medium
with mlflow.start_run(run_name="mipro-medium"):
    mlflow.log_param("optimizer", "MIPROv2")
    mlflow.log_param("auto", "medium")

    program = dspy.ChainOfThought("question -> answer")
    optimizer = dspy.MIPROv2(metric=metric, auto="medium")
    optimized = optimizer.compile(program, trainset=trainset)

    score = Evaluate(devset=devset, metric=metric, num_threads=4)(optimized)
    mlflow.log_metric("dev_score", score)

    optimized.save("mipro_medium_optimized.json")
    mlflow.log_artifact("mipro_medium_optimized.json")
```

### Compare in the MLflow UI

1. Open `http://localhost:5000`
2. Click "qa-optimization" experiment
3. Select all 3 runs → click "Compare"
4. View the metrics chart to see which run scored highest
5. Click the winning run to download its artifact

## Register the best model and load in production

```python
import mlflow

# Register the best experiment's model
with mlflow.start_run(run_name="register-best"):
    program = dspy.ChainOfThought("question -> answer")
    program.load("mipro_medium_optimized.json")

    mlflow.dspy.log_model(program, "qa-model")
    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/qa-model",
        "production-qa"
    )

# In production code
model = mlflow.dspy.load_model("models:/production-qa/latest")

# Use in a FastAPI endpoint
from fastapi import FastAPI
app = FastAPI()

@app.post("/query")
async def query(question: str):
    result = model(question=question)
    return {"answer": result.answer}
```
