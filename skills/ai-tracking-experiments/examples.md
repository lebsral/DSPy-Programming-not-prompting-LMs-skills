# Experiment Tracking Examples

## Example 1: Comparing 5 optimizer configs for a classification task

You're building a ticket classifier and want to find the best optimization approach.

### Setup

```python
import dspy
from dspy.evaluate import Evaluate

# The program
class TicketClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(
            "ticket_text -> category: Literal['bug', 'feature', 'question', 'billing']"
        )

    def forward(self, ticket_text):
        return self.classify(ticket_text=ticket_text)

# Metric
def accuracy(example, prediction, trace=None):
    return prediction.category.lower() == example.category.lower()

# Data (200 train, 50 dev)
trainset = [
    dspy.Example(ticket_text="App crashes on login", category="bug").with_inputs("ticket_text"),
    dspy.Example(ticket_text="Can you add dark mode?", category="feature").with_inputs("ticket_text"),
    # ... 200 examples
]
devset = trainset[150:]   # hold out 50 for evaluation
trainset = trainset[:150]
```

### Run all experiments

```python
experiments = [
    {
        "name": "baseline-no-opt",
        "optimizer_class": None,
        "optimizer_kwargs": {},
    },
    {
        "name": "bootstrap-4",
        "optimizer_class": dspy.BootstrapFewShot,
        "optimizer_kwargs": {"metric": accuracy, "max_bootstrapped_demos": 4},
    },
    {
        "name": "bootstrap-8",
        "optimizer_class": dspy.BootstrapFewShot,
        "optimizer_kwargs": {"metric": accuracy, "max_bootstrapped_demos": 8},
    },
    {
        "name": "mipro-light",
        "optimizer_class": dspy.MIPROv2,
        "optimizer_kwargs": {"metric": accuracy, "auto": "light"},
    },
    {
        "name": "mipro-medium",
        "optimizer_class": dspy.MIPROv2,
        "optimizer_kwargs": {"metric": accuracy, "auto": "medium"},
    },
]

for exp in experiments:
    if exp["optimizer_class"] is None:
        # Just evaluate baseline
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)
        evaluator = Evaluate(devset=devset, metric=accuracy, num_threads=4)
        score = evaluator(TicketClassifier())
        log_experiment({
            "name": exp["name"],
            "optimizer": "none",
            "model": "openai/gpt-4o-mini",
            "score": score,
            "baseline_score": score,
            "improvement": 0,
            "cost_usd": 0.02,
            "artifact_path": None,
        })
    else:
        run_experiment(
            name=exp["name"],
            program_class=TicketClassifier,
            optimizer_class=exp["optimizer_class"],
            optimizer_kwargs=exp["optimizer_kwargs"],
            trainset=trainset,
            devset=devset,
            metric=accuracy,
        )
```

### Compare results

```python
compare_experiments()
# Name                           Optimizer            Model                   Score  Improve    Cost
# ------------------------------------------------------------------------------------------------------------------------
# mipro-medium                   MIPROv2              openai/gpt-4o-mini       91.0%   +23.0%  $5.80
# mipro-light                    MIPROv2              openai/gpt-4o-mini       86.0%   +18.0%  $1.50
# bootstrap-8                    BootstrapFewShot     openai/gpt-4o-mini       82.0%   +14.0%  $0.40
# bootstrap-4                    BootstrapFewShot     openai/gpt-4o-mini       78.0%   +10.0%  $0.20
# baseline-no-opt                none                 openai/gpt-4o-mini       68.0%    +0.0%  $0.02

# Decision: mipro-medium wins at 91% accuracy
promote_experiment("mipro-medium")
```

## Example 2: Model migration experiment (GPT-4o to Claude Sonnet)

You're considering switching models and need to know the impact.

### Run the same optimizer on both models

```python
models = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4-5-20250929",
]

for model in models:
    # Baseline (no optimization)
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    evaluator = Evaluate(devset=devset, metric=accuracy, num_threads=4)
    baseline = evaluator(TicketClassifier())

    log_experiment({
        "name": f"baseline-{model.split('/')[-1]}",
        "optimizer": "none",
        "model": model,
        "score": baseline,
        "baseline_score": baseline,
        "improvement": 0,
    })

    # MIPROv2 optimization
    run_experiment(
        name=f"mipro-{model.split('/')[-1]}",
        program_class=TicketClassifier,
        optimizer_class=dspy.MIPROv2,
        optimizer_kwargs={"metric": accuracy, "auto": "medium"},
        trainset=trainset,
        devset=devset,
        metric=accuracy,
        model=model,
    )

compare_experiments()
# Name                           Optimizer            Model                   Score  Improve    Cost
# ------------------------------------------------------------------------------------------------------------------------
# mipro-gpt-4o                   MIPROv2              openai/gpt-4o            94.0%   +14.0% $12.00
# mipro-claude-sonnet-4-5-20250929   MIPROv2              anthropic/claude-...     93.0%   +18.0%  $8.50
# mipro-gpt-4o-mini              MIPROv2              openai/gpt-4o-mini       91.0%   +23.0%  $5.80
# baseline-gpt-4o                none                 openai/gpt-4o            80.0%    +0.0%  $0.00
# baseline-claude-sonnet-4-5-20250929   none                 anthropic/claude-...     75.0%    +0.0%  $0.00
# baseline-gpt-4o-mini           none                 openai/gpt-4o-mini       68.0%    +0.0%  $0.00
```

### Analyze the results

```python
# GPT-4o gets best absolute score (94%) but is most expensive
# Claude Sonnet gets similar score (93%) at lower cost
# GPT-4o-mini + MIPROv2 (91%) is the best value — close to GPT-4o at 1/2 the cost

# Check cost per point of accuracy
runs = load_experiments()
for r in runs:
    if r.get("optimizer") != "none" and r.get("cost_usd", 0) > 0:
        cost_per_point = r["cost_usd"] / r["improvement"] if r["improvement"] > 0 else float("inf")
        print(f"{r['name']}: ${cost_per_point:.2f} per point of improvement")

# mipro-gpt-4o: $0.86 per point
# mipro-claude-sonnet-4-5-20250929: $0.47 per point
# mipro-gpt-4o-mini: $0.25 per point  <-- best value
```

### Stacked optimization (advanced)

Run BootstrapFewShot first, then MIPROv2 on the result:

```python
# Stage 1: BootstrapFewShot
optimized_stage1, run1 = run_experiment(
    name="stacked-stage1-bootstrap",
    program_class=TicketClassifier,
    optimizer_class=dspy.BootstrapFewShot,
    optimizer_kwargs={"metric": accuracy, "max_bootstrapped_demos": 4},
    trainset=trainset,
    devset=devset,
    metric=accuracy,
)

# Stage 2: MIPROv2 on top of BootstrapFewShot result
def make_preoptimized():
    prog = TicketClassifier()
    prog.load("artifacts/stacked-stage1-bootstrap.json")
    return prog

optimized_stage2, run2 = run_experiment(
    name="stacked-stage2-mipro",
    program_class=make_preoptimized,
    optimizer_class=dspy.MIPROv2,
    optimizer_kwargs={"metric": accuracy, "auto": "medium"},
    trainset=trainset,
    devset=devset,
    metric=accuracy,
)

print(f"Bootstrap alone: {run1['score']:.1f}%")
print(f"Bootstrap + MIPROv2: {run2['score']:.1f}%")
# Bootstrap alone: 78.0%
# Bootstrap + MIPROv2: 92.5%  <-- stacking helps!
```
