# DSPy Project Setup Reference

> Condensed from [dspy.ai/api/](https://dspy.ai/api/) and [dspy.ai/learn/](https://dspy.ai/learn/). Verify against upstream for latest. DSPy 3.2.1+.

## Installation

```bash
pip install -U dspy
```

DSPy requires Python 3.10+. Minimum version in `requirements.txt`: `dspy>=3.0`

Add extras as needed:

| Need | Package(s) |
|------|-----------|
| Load HuggingFace datasets | `datasets` |
| Serve as a web API | `fastapi uvicorn[standard] pydantic-settings>=2.0` |
| Vector search | provider-specific (`pinecone-client`, `chromadb`, etc.) |

## Project structure

```
my-ai-feature/
├── main.py          # Entry point — run your AI feature
├── program.py       # AI logic (DSPy module)
├── metrics.py       # How to measure if the AI is working
├── optimize.py      # Automatically improve prompts
├── evaluate.py      # Test quality on held-out data
├── data.py          # Training/eval data loading
└── requirements.txt # Dependencies
```

## LM configuration

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

`dspy.LM` accepts any [LiteLLM-supported model string](https://docs.litellm.ai/docs/providers):

| Provider | Model string |
|----------|-------------|
| OpenAI | `"openai/gpt-4o-mini"` |
| Anthropic | `"anthropic/claude-sonnet-4-5-20250929"` |
| Azure OpenAI | `"azure/gpt-4o"` |
| Ollama (local) | `"ollama/llama3"` |

The LM reads its API key from the provider's standard environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). Use a `.env` file or your platform's secret manager — never hardcode keys.

### `dspy.LM` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | (required) | LiteLLM model string, e.g. `"openai/gpt-4o-mini"` |
| `model_type` | str | `'chat'` | API call type: `'chat'`, `'text'`, or `'responses'` |
| `temperature` | float | None | Generation temperature; None uses the LM default |
| `max_tokens` | int | None | Max output tokens; None uses the LM default |
| `cache` | bool | `True` | Cache responses for reuse |
| `num_retries` | int | `3` | Retries on transient failure |

## Module selection

| Task | Module | Notes |
|------|--------|-------|
| Simple extraction, lookup, yes/no | `dspy.Predict` | Lowest cost; no reasoning step |
| Most tasks | `dspy.ChainOfThought` | Default choice; adds `reasoning` output automatically |
| Math, counting, dates | `dspy.ProgramOfThought` | Generates and executes code to compute the answer |
| Needs tool calls | `dspy.ReAct` | Reasoning + action loop; pass a `tools` list |

Do not add a `reasoning` field to your signature when using `ChainOfThought` — DSPy injects it automatically.

## Signature pattern

```python
class MySignature(dspy.Signature):
    """Task instruction goes here (becomes the system prompt)."""
    input_field: str = dspy.InputField(desc="what this contains")
    output_field: str = dspy.OutputField(desc="what this should be")
```

For constrained outputs (categories, labels), use `Literal[tuple(LIST)]` — not `str`:

```python
from typing import Literal

LABELS = ["billing", "technical", "account"]

class Route(dspy.Signature):
    message: str = dspy.InputField()
    team: Literal[tuple(LABELS)] = dspy.OutputField()
```

## dspy.Example and .with_inputs()

Every example in training and eval data must call `.with_inputs()` to declare which fields are inputs:

```python
example = dspy.Example(email="I was charged twice.", team="billing").with_inputs("email")
```

Without `.with_inputs()`, the optimizer cannot distinguish inputs from expected outputs and produces incorrect demos.

## Evaluate

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
score = evaluator(program)   # returns a float (percentage)
```

Always run `evaluate.py` to get a baseline score before running `optimize.py`.

## Optimizer selection

| Optimizer | Data needed | Best for |
|-----------|-------------|---------|
| `dspy.BootstrapFewShot` | 20-50 examples | Quick start; first optimization |
| `dspy.MIPROv2` | 200+ examples | Best prompt and instruction optimization |
| `dspy.BootstrapFinetune` | 500+ examples | Maximum quality on smaller LMs |

```python
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(program, trainset=trainset)
```

### `dspy.BootstrapFewShot` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | Callable | None | Scoring function — compares expected vs predicted |
| `max_bootstrapped_demos` | int | `4` | Max bootstrapped demonstrations per predictor |
| `max_labeled_demos` | int | `16` | Max labeled demonstrations per predictor |
| `max_rounds` | int | `1` | Bootstrap attempts per training example |
| `metric_threshold` | float | None | Threshold for numerical metric results |

### `dspy.Evaluate` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | list[Example] | (required) | Evaluation dataset |
| `metric` | Callable | (required) | Scoring function |
| `num_threads` | int | `1` | Parallel evaluation threads |
| `display_progress` | bool | `False` | Show progress bar |

## Save and load

```python
# After optimization
optimized.save("optimized.json")

# In main.py
program = MyProgram()
program.load("optimized.json")
```

## Inspecting LM calls

```python
dspy.inspect_history(n=3)  # prints the last 3 LM prompts and responses
```

## Kickoff checklist

- [ ] `dspy.LM` configured and API key set in environment
- [ ] Signature field names match your actual task — not the generic `input_field`/`output_field` placeholders
- [ ] All `dspy.Example` objects call `.with_inputs()` with the input field name(s)
- [ ] Metric function returns `True`/`False` or a `float` between `0.0` and `1.0`
- [ ] Baseline score measured with `evaluate.py` before running `optimize.py`
- [ ] Optimized program saved to `optimized.json` and loaded in `main.py`
