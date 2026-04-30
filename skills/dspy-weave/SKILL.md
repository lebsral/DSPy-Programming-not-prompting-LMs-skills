---
name: dspy-weave
description: Use W&B Weave for DSPy experiment tracking and observability. Use when you want to set up Weave, W&B, wandb, Weights & Biases, experiment dashboard, weave.op, or team collaboration for DSPy. Also: weave setup, pip install weave, weave.init, wandb project, W&B experiment tracking, weave decorator, weave.op decorator, wandb dashboard, compare optimization runs, team experiment tracking.
---

# W&B Weave — Cloud Observability & Experiment Tracking for DSPy

Guide the user through setting up W&B Weave for tracing DSPy calls, tracking optimization experiments, and collaborating with team dashboards.

## What is W&B Weave

Weave is Weights & Biases' LLM observability and experiment tracking product. It provides cloud-hosted dashboards for tracing function calls, comparing optimization runs, and sharing results across teams.

- **Cloud-hosted**: Dashboards at [wandb.ai](https://wandb.ai)
- **Manual instrumentation**: Uses `@weave.op()` decorator (not auto-instrument like Langtrace)
- **Team collaboration**: shared projects, comments, and run comparisons

### Key difference from Langtrace/Phoenix

Weave uses a **manual decorator** (`@weave.op()`) — you choose which functions to trace. Langtrace and Phoenix auto-instrument all DSPy calls. This gives Weave more control over what gets tracked but requires more setup.

## When to use Weave

Use Weave when:

- Your team already uses W&B for ML experiments
- You want cloud-hosted dashboards with team collaboration
- You want to track and compare optimization runs side-by-side
- You need fine-grained control over which functions are traced

Do NOT use Weave when:

- You want auto-instrumentation with zero code changes — see `/dspy-langtrace`
- You want a free, local-only trace viewer — see `/dspy-phoenix`
- You need the full ML lifecycle (model registry, deployment) — see `/dspy-mlflow`
- You're a solo developer who doesn't need team features — Langtrace or Phoenix is simpler

## Setup

### Install

```bash
pip install weave
```

### Initialize

```python
import weave

weave.init("my-dspy-project")  # Creates project at wandb.ai

# You'll be prompted to log in on first run
# Or set WANDB_API_KEY environment variable
```

### Environment variable configuration

```bash
export WANDB_API_KEY="your-key"          # From wandb.ai/settings
export WANDB_ENTITY="your-team"          # Optional: team name
export WANDB_PROJECT="my-dspy-project"   # Optional: project name
```

## Tracing with @weave.op()

The `@weave.op()` decorator traces a function's inputs, outputs, latency, and cost:

```python
import weave
import dspy

weave.init("my-dspy-project")
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

class QABot(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)

bot = QABot()

@weave.op()
def handle_question(question: str) -> str:
    """Traced by Weave — inputs, outputs, and latency logged."""
    result = bot(question=question)
    return result.answer

# Every call is tracked at wandb.ai
answer = handle_question("How do refunds work?")
```

### Tracing multiple functions

Decorate each function you want to trace:

```python
@weave.op()
def retrieve_context(question: str) -> list[str]:
    return dspy.Retrieve(k=3)(question).passages

@weave.op()
def generate_answer(context: list[str], question: str) -> str:
    cot = dspy.ChainOfThought("context, question -> answer")
    return cot(context=context, question=question).answer

@weave.op()
def handle_question(question: str) -> str:
    context = retrieve_context(question)
    return generate_answer(context, question)

# Weave shows the call tree: handle_question -> retrieve_context, generate_answer
```

## Tracking optimization experiments

Weave excels at comparing optimization runs. Wrap your optimization in `@weave.op()`:

```python
import weave
import dspy
from dspy.evaluate import Evaluate

weave.init("optimization-experiments")
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

@weave.op()
def run_optimization(optimizer_name: str, model: str, auto_setting: str):
    """Run and track an optimization experiment."""
    lm = dspy.LM(model)
    dspy.configure(lm=lm)

    program = dspy.ChainOfThought("question -> answer")

    if optimizer_name == "miprov2":
        optimizer = dspy.MIPROv2(metric=metric, auto=auto_setting)
    elif optimizer_name == "bootstrap":
        optimizer = dspy.BootstrapFewShot(metric=metric)

    optimized = optimizer.compile(program, trainset=trainset)

    evaluator = Evaluate(devset=devset, metric=metric, num_threads=4)
    score = evaluator(optimized)

    # Save the artifact
    path = f"experiments/{optimizer_name}_{model}_{auto_setting}.json"
    optimized.save(path)

    return {
        "score": score,
        "optimizer": optimizer_name,
        "model": model,
        "auto": auto_setting,
        "artifact_path": path,
    }

# Run experiments — each is tracked in Weave
run_optimization("miprov2", "openai/gpt-4o-mini", "light")
run_optimization("miprov2", "openai/gpt-4o-mini", "medium")
run_optimization("bootstrap", "openai/gpt-4o-mini", "n/a")
```

### Comparing runs in the W&B dashboard

1. Go to [wandb.ai](https://wandb.ai) and open your project
2. Click on the "Traces" tab to see all tracked calls
3. Compare inputs and outputs across runs
4. Sort by score to find the best experiment
5. Share the dashboard URL with your team

## Weave vs Langtrace vs Phoenix

| Feature | W&B Weave | Langtrace | Arize Phoenix |
|---------|-----------|-----------|---------------|
| Instrumentation | Manual (`@weave.op()`) | Auto (one line) | Auto (plugin) |
| Setup effort | Decorator per function | One line | Two lines + launch |
| Cloud dashboard | Yes (wandb.ai) | Yes (app.langtrace.ai) | Yes (Arize platform) |
| Local/self-hosted | No | Yes (Docker) | Yes (`px.launch_app()`) |
| Team collaboration | Yes (built-in) | Basic | Basic |
| Experiment comparison | Yes (side-by-side) | No | No |
| Built-in evals | Basic | Basic | Yes (evals module) |
| Cost | Free tier + paid plans | Free tier + paid | Free (open source) |
| Best for | Teams on W&B, experiment tracking | DSPy-first auto-tracing | Local trace viewer + evals |

### Decision guide

```
Want DSPy observability?
|
+- Team already uses W&B? -> Weave
+- Want auto-instrumentation (no decorators)? -> Langtrace (/dspy-langtrace)
+- Want local-only + built-in evals? -> Phoenix (/dspy-phoenix)
+- Need full ML lifecycle (registry, deploy)? -> MLflow (/dspy-mlflow)
```

## Verifying the setup

After initializing Weave and adding `@weave.op()` decorators, run one traced call and confirm it appears in the dashboard:

```python
# Quick smoke test
@weave.op()
def smoke_test(x: str) -> str:
    return x.upper()

result = smoke_test("hello")
print(f"Check your project at https://wandb.ai — look for the smoke_test call")
```

If the call does not appear: check `WANDB_API_KEY` is set, confirm `weave.init()` was called before the decorated function, and verify network access to wandb.ai.

## Gotchas

- **Claude puts `@weave.op()` on the DSPy module class instead of the calling function.** Weave decorators trace regular functions, not DSPy module classes. Decorate the function that *calls* the module, not the module itself. `@weave.op()` goes on `handle_question()`, not on `QABot`.
- **Claude calls `weave.init()` inside a function instead of at module level.** `weave.init()` must run once at startup, before any `@weave.op()` decorated functions are called. Placing it inside a request handler creates a new project per call and fragments your traces.
- **Claude forgets to set `WANDB_API_KEY` in deployment environments.** Local development prompts for login interactively, but production (Docker, CI, serverless) needs the environment variable explicitly set. Always include `WANDB_API_KEY` in environment configuration for non-local setups.
- **Claude auto-instruments everything instead of using selective decorators.** Unlike Langtrace/Phoenix, Weave traces only what you decorate. Claude sometimes tries to add a global "trace all DSPy calls" setup that does not exist. Each function needs its own `@weave.op()` decorator.
- **Claude nests `@weave.op()` and DSPy decorators incorrectly.** If combining with other decorators, `@weave.op()` should be the outermost decorator so it captures the full function execution including any inner decorator behavior.

## Additional resources

- [W&B Weave docs](https://docs.wandb.ai/weave/)
- [weave.op() reference](https://docs.wandb.ai/weave/reference/python-sdk/weave/)
- [W&B dashboard](https://wandb.ai)
- For API details, see [reference.md](reference.md)
- For worked examples, see [examples.md](examples.md)

## Cross-references

- **Langtrace** (auto-instrumentation, easiest setup) — `/dspy-langtrace`
- **Arize Phoenix** (open-source with evals) — `/dspy-phoenix`
- **MLflow** (full ML lifecycle) — `/dspy-mlflow`
- **Aggregate monitoring** — `/ai-monitoring`
- **Experiment tracking patterns** (JSONL-based, lightweight) — `/ai-tracking-experiments`
- Not sure which skill to use next? Try `/ai-do` to get routed to the right one
