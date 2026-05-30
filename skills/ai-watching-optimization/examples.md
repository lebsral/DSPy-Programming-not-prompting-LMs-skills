# Watching Optimization Examples

## Example 1: Built-in GEPA monitoring with track_stats

Monitor a GEPA optimization run using the built-in `track_stats=True` flag, with baseline comparison before and after.

```python
import dspy
from dspy.evaluate import Evaluate

# Configure LMs
task_lm = dspy.LM("openai/gpt-4o-mini")
reflection_lm = dspy.LM("openai/gpt-4o", temperature=1.0, max_tokens=4096)
dspy.configure(lm=task_lm)

# Define a simple classifier
class TicketRouter(dspy.Signature):
    """Route a support ticket to the correct team."""
    ticket: str = dspy.InputField(desc="Support ticket text")
    team: str = dspy.OutputField(desc="Team name: billing, technical, account, other")

program = dspy.ChainOfThought(TicketRouter)

# Prepare data
trainset = [
    dspy.Example(ticket="I was charged twice for my subscription", team="billing").with_inputs("ticket"),
    dspy.Example(ticket="App crashes when I click settings", team="technical").with_inputs("ticket"),
    # ... 30-50 more examples
]
valset = trainset[40:]  # Hold out some for validation
trainset = trainset[:40]

# Define metric
def routing_accuracy(example, pred, trace=None):
    return example.team.lower().strip() == pred.team.lower().strip()

# Step 1: Baseline evaluation
evaluator = Evaluate(devset=valset, metric=routing_accuracy, num_threads=8)
baseline_score = evaluator(program)
print(f"Baseline: {baseline_score}")

# Step 2: Optimize with tracking enabled
optimizer = dspy.GEPA(
    metric=routing_accuracy,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
    track_stats=True,
)

optimized = optimizer.compile(program, trainset=trainset)

# Step 3: Inspect iteration-by-iteration progress
stats = optimizer.detailed_results
for i, result in enumerate(stats):
    print(f"Iteration {i}: score={result['score']:.3f}")

# Step 4: Final evaluation
optimized_score = evaluator(optimized)
print(f"\nBaseline:   {baseline_score}")
print(f"Optimized:  {optimized_score}")
print(f"Improvement: {optimized_score - baseline_score:+.1f}")

# Step 5: Check for overfitting
train_eval = Evaluate(devset=trainset, metric=routing_accuracy, num_threads=8)
train_score = train_eval(optimized)
if train_score - optimized_score > 10:
    print("Warning: possible overfitting (train >> val)")
```

## Example 2: LangWatch with MIPROv2

Watch a MIPROv2 optimization run in real time via the LangWatch cloud dashboard.

```bash
pip install langwatch
```

```python
import langwatch
import dspy
from dspy.evaluate import Evaluate

# Configure
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Initialize LangWatch -- streams optimizer progress to dashboard
langwatch.dspy.init(experiment="ticket-router-miprov2")

# Define program and metric (same as Example 1)
class TicketRouter(dspy.Signature):
    """Route a support ticket to the correct team."""
    ticket: str = dspy.InputField(desc="Support ticket text")
    team: str = dspy.OutputField(desc="Team name: billing, technical, account, other")

program = dspy.ChainOfThought(TicketRouter)

def routing_accuracy(example, pred, trace=None):
    return example.team.lower().strip() == pred.team.lower().strip()

# Baseline
evaluator = Evaluate(devset=valset, metric=routing_accuracy, num_threads=8)
baseline = evaluator(program)
print(f"Baseline: {baseline}")

# Optimize -- LangWatch streams progress to app.langwatch.ai
optimizer = dspy.MIPROv2(metric=routing_accuracy, auto="light")
optimized = optimizer.compile(program, trainset=trainset)

# Final evaluation
final = evaluator(optimized)
print(f"Optimized: {final} (was {baseline})")

# Now go to app.langwatch.ai to see:
# - Live score chart showing each candidate's score
# - Cost accumulation over the optimization run
# - Current predictor states (instructions + demos)
# - Compare this run against previous experiments
```

**What to look for in the dashboard:**
- Scores should trend upward. If flat from the start, check your metric.
- Cost chart shows total spend. Compare cost-per-point across iterations.
- Predictor states show exactly what instructions MIPROv2 is trying.
- If scores plateau early, consider `auto="medium"` for more exploration.

## Example 3: Custom BaseCallback for BootstrapFewShot

Print live progress during a BootstrapFewShot optimization. Works with any optimizer.

```python
import dspy
from dspy.evaluate import Evaluate

class LiveProgressCallback(dspy.BaseCallback):
    """Print evaluation results as they happen during optimization."""
    def __init__(self):
        super().__init__()
        self.eval_count = 0
        self.best_score = 0.0

    def on_evaluate_end(self, instance, inputs, outputs, exception):
        self.eval_count += 1
        score = outputs.get("score", None)
        if score is not None:
            if score > self.best_score:
                self.best_score = score
                marker = " <-- new best!"
            else:
                marker = ""
            print(f"[Eval {self.eval_count:3d}] Score: {score:.3f} (best: {self.best_score:.3f}){marker}")

# Register callback
progress = LiveProgressCallback()
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    callbacks=[progress],
)

# Define program
class TicketRouter(dspy.Signature):
    """Route a support ticket to the correct team."""
    ticket: str = dspy.InputField(desc="Support ticket text")
    team: str = dspy.OutputField(desc="Team name: billing, technical, account, other")

program = dspy.ChainOfThought(TicketRouter)

def routing_accuracy(example, pred, trace=None):
    return example.team.lower().strip() == pred.team.lower().strip()

# Baseline
evaluator = Evaluate(devset=valset, metric=routing_accuracy, num_threads=8)
baseline = evaluator(program)
print(f"\nBaseline: {baseline}\n")
print("Starting optimization...")
print("-" * 50)

# Reset counter for optimization tracking
progress.eval_count = 0
progress.best_score = 0.0

# Optimize -- callback prints progress automatically
optimizer = dspy.BootstrapFewShot(
    metric=routing_accuracy,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)
optimized = optimizer.compile(program, trainset=trainset)

print("-" * 50)
final = evaluator(optimized)
print(f"\nBaseline:  {baseline}")
print(f"Optimized: {final}")
print(f"Total evaluations during optimization: {progress.eval_count}")
```

**Output looks like:**
```
Baseline: 72.0

Starting optimization...
--------------------------------------------------
[Eval   1] Score: 0.720 (best: 0.720) <-- new best!
[Eval   2] Score: 0.680 (best: 0.720)
[Eval   3] Score: 0.760 (best: 0.760) <-- new best!
[Eval   4] Score: 0.740 (best: 0.760)
[Eval   5] Score: 0.800 (best: 0.800) <-- new best!
[Eval   6] Score: 0.800 (best: 0.800)
[Eval   7] Score: 0.780 (best: 0.800)
--------------------------------------------------

Baseline:  72.0
Optimized: 80.0
Total evaluations during optimization: 7
```
