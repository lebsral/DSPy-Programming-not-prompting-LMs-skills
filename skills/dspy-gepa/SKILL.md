---
name: dspy-gepa
description: Use when you want to optimize instructions without few-shot examples — a lightweight alternative to COPRO when you do not have or do not want to use demonstrations. Common scenarios - optimizing instructions when you do not have or do not want to use few-shot demonstrations, lightweight instruction search as a first step, tasks where examples in the prompt confuse the model, or when you want fast instruction optimization without the cost of COPRO. Related - ai-improving-accuracy, dspy-copro, dspy-miprov2. Also used for dspy.GEPA, instruction optimization without demos, lightweight prompt optimization, optimize instructions only, no few-shot examples needed, GEPA vs COPRO, quick instruction search, when demonstrations hurt performance, zero-shot optimization, instruction-only optimizer, simplest instruction tuner, fast prompt optimization, skip few-shot and just tune instructions, optimize Pydantic field descriptions, GEPA structured output, GEPA does not optimize field desc.
---

# Instruction Optimization with dspy.GEPA

Guide the user through using `dspy.GEPA` to automatically discover better instructions for their DSPy programs through reflective evolution.

## What is dspy.GEPA

`dspy.GEPA` is a DSPy optimizer that evolves the **instruction text** in your program's predictors. Rather than adding few-shot examples (like BootstrapFewShot) or tuning model weights (like BootstrapFinetune), GEPA iteratively proposes, evaluates, and refines the natural-language instructions that guide each LM call.

Key properties:

- **Tunes instructions only** -- no few-shot demos are injected into prompts, keeping them compact
- **Uses textual feedback** -- a reflection LM reads execution traces and failure feedback to propose better instructions, not just scalar scores
- **Maintains a Pareto frontier** -- tracks multiple candidate programs that excel on different subsets, then merges the best traits
- **Works with ~50 examples** -- needs less data than MIPROv2 or BootstrapFewShotWithRandomSearch
- **Supports per-predictor feedback** -- metrics can return targeted feedback for individual predictors in multi-step pipelines

## When to use GEPA

Use `dspy.GEPA` when:

- You have ~50+ labeled examples (fewer than what MIPROv2 needs to shine)
- You want to optimize **instructions** without adding few-shot examples to the prompt
- Your task has interpretable failure modes you can describe in natural language
- You have a multi-step pipeline and want per-predictor instruction tuning
- You want compact prompts (no demo bloat) while still improving quality

Do **not** use GEPA when:

- You have no way to provide textual feedback on failures -- use `dspy.BootstrapFewShot` instead
- You need the best possible prompt optimization and have 200+ examples -- use `dspy.MIPROv2`
- You want to tune model weights -- use `dspy.BootstrapFinetune`
- Your task is trivially solved without instruction tuning -- use `dspy.Predict` or `dspy.ChainOfThought` directly

## Basic usage

Three things are needed: a DSPy program, a feedback metric, and a training set.

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

# 1. Define your program
classify = dspy.ChainOfThought("text -> label")

# 2. Define a feedback metric
# GEPA metrics can return a float OR a dict with score + feedback text
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = float(pred.label == gold.label)
    feedback = "" if score == 1.0 else f"Expected '{gold.label}', got '{pred.label}'."
    return {"score": score, "feedback": feedback}

# 3. Prepare training data
trainset = [
    dspy.Example(text="Great product!", label="positive").with_inputs("text"),
    dspy.Example(text="Terrible service.", label="negative").with_inputs("text"),
    # ... ~50 examples
]

# 4. Optimize
gepa = dspy.GEPA(
    metric=metric,
    reflection_lm=dspy.LM("openai/gpt-4o", temperature=1.0, max_tokens=4096),  # use a strong model for reflection
    auto="light",
)
optimized = gepa.compile(classify, trainset=trainset)

# 5. Use the optimized program
result = optimized(text="This exceeded my expectations!")
print(result.label)

# 6. Save for later
optimized.save("optimized_classifier.json")
```

## Constructor parameters

```python
dspy.GEPA(
    metric,                              # GEPAFeedbackMetric (required)
    *,
    auto=None,                           # "light", "medium", or "heavy"
    max_full_evals=None,                 # int -- full validation passes allowed
    max_metric_calls=None,               # int -- total metric invocations allowed
    reflection_lm=None,                  # LM for proposing new instructions
    reflection_minibatch_size=3,         # examples per reflection step
    candidate_selection_strategy="pareto",  # "pareto" or "current_best"
    skip_perfect_score=True,             # skip examples already scoring perfectly
    add_format_failure_as_feedback=False, # include format errors in feedback
    instruction_proposer=None,           # custom proposal function
    component_selector="round_robin",    # which predictor to improve next
    use_merge=True,                      # merge successful variants
    max_merge_invocations=5,             # merge attempt limit
    num_threads=None,                    # parallel evaluation threads
    failure_score=0.0,                   # score for failed examples
    perfect_score=1.0,                   # score that counts as perfect
    log_dir=None,                        # directory for optimization logs
    track_stats=False,                   # return detailed metadata
    track_best_outputs=False,            # retain best outputs per task
    seed=0,                              # reproducibility seed
)
```

### Key parameters explained

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `metric` | required | Feedback function -- returns float or `{"score": float, "feedback": str}` |
| `auto` | None | Budget preset: `"light"` (fast), `"medium"` (balanced), `"heavy"` (thorough) |
| `reflection_lm` | None | LM that proposes new instructions. Use a strong model (e.g., GPT-4o, Claude Sonnet). Required unless you provide a custom `instruction_proposer` |
| `reflection_minibatch_size` | 3 | How many examples the reflection LM sees per iteration. Larger = better proposals but more cost |
| `candidate_selection_strategy` | `"pareto"` | `"pareto"` maintains diverse candidates; `"current_best"` always mutates the top scorer |
| `use_merge` | True | After evolving candidates, merge the best modules from different lineages |
| `max_merge_invocations` | 5 | Cap on merge attempts to control cost |
| `skip_perfect_score` | True | Do not waste budget on examples already scoring `perfect_score` |
| `track_stats` | False | When True, attach optimization metadata to `optimized.detailed_results` |

### Budget control

Exactly **one** of these three must be set:

- **`auto`** -- preset budget (`"light"`, `"medium"`, `"heavy"`)
- **`max_full_evals`** -- number of full passes over the validation set
- **`max_metric_calls`** -- total number of metric invocations

Start with `auto="light"` for quick experiments, then move to `"medium"` or `"heavy"` for production.

## Writing feedback metrics

GEPA metrics are more expressive than standard DSPy metrics. They accept additional keyword arguments for trace-level feedback:

```python
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Args:
        gold: The expected Example (ground truth)
        pred: The model's Prediction
        trace: Full program execution trace (optional)
        pred_name: Name of the predictor being optimized (optional)
        pred_trace: Sub-trace for just this predictor (optional)

    Returns:
        float -- simple score
        OR dict -- {"score": float, "feedback": str}
    """
```

### Returning textual feedback

The key advantage of GEPA over other optimizers is that metrics can explain **why** an output failed. The reflection LM reads this feedback to propose better instructions.

```python
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    if pred.answer == gold.answer:
        return {"score": 1.0, "feedback": ""}

    feedback_parts = []
    if len(pred.answer) > 200:
        feedback_parts.append("Answer is too verbose. Keep it under 200 characters.")
    if gold.answer.lower() not in pred.answer.lower():
        feedback_parts.append(f"Answer should contain '{gold.answer}'.")

    return {
        "score": 0.0,
        "feedback": " ".join(feedback_parts),
    }
```

Write feedback that is **actionable** -- describe what the instruction should encourage or discourage. Vague feedback like "wrong answer" does not help the reflection LM.

## How GEPA works internally

Understanding the algorithm helps you write better metrics and choose parameters:

1. **Initialize** -- seeds the candidate pool with the unoptimized program
2. **Select candidate** -- picks a program from the Pareto frontier (diverse strengths)
3. **Sample minibatch** -- draws `reflection_minibatch_size` examples from trainset
4. **Collect traces and feedback** -- runs the candidate, captures execution traces and metric feedback
5. **Select component** -- picks which predictor to improve (round-robin by default)
6. **Reflect and mutate** -- the `reflection_lm` reads traces + feedback and proposes a revised instruction
7. **Evaluate** -- scores the new candidate on the minibatch; if promising, validates on the full set
8. **Update frontier** -- adds the candidate to the Pareto frontier if it is non-dominated
9. **Merge** -- combines the best predictors from different candidate lineages into one program
10. **Terminate** -- returns the best aggregate performer when the budget is exhausted

The Pareto frontier is the key innovation: rather than keeping only the single best candidate, GEPA maintains candidates that excel on different subsets. This prevents the optimizer from overfitting to one failure pattern while ignoring others.

## GEPA vs MIPROv2 -- when to use which

| Aspect | `dspy.GEPA` | `dspy.MIPROv2` |
|--------|-------------|----------------|
| **What it tunes** | Instructions only | Instructions + few-shot demos |
| **Data needed** | ~50 examples | ~200 examples |
| **Prompt size** | Compact (no demos) | Larger (includes demos) |
| **Feedback** | Uses textual feedback from metrics | Uses scalar scores only |
| **Multi-step** | Per-predictor feedback and optimization | Optimizes all predictors jointly |
| **Best for** | Instruction tuning, compact prompts, rich feedback | Maximum prompt quality, larger budgets |
| **Cost** | Lower (fewer metric calls) | Higher (explores more candidates) |

**Rule of thumb:** Start with GEPA if you have fewer than 200 examples or want compact prompts. Move to MIPROv2 if you have more data and are willing to trade prompt size for quality.

## Providing a validation set

If you have a separate validation set, pass it to `compile`:

```python
optimized = gepa.compile(
    classify,
    trainset=trainset,
    valset=valset,
)
```

Without a `valset`, GEPA uses the trainset for both training and validation. This can lead to overfitting but is useful for test-time search (optimizing for a specific batch of inputs).

## Inference-time search

GEPA can be used at inference time to find the best instructions for a specific batch of tasks:

```python
gepa = dspy.GEPA(
    metric=metric,
    reflection_lm=dspy.LM("openai/gpt-4o", temperature=1.0, max_tokens=4096),  # use a strong model for reflection
    auto="light",
    track_stats=True,
    track_best_outputs=True,
)

# Pass the same data as both trainset and valset
result = gepa.compile(program, trainset=tasks, valset=tasks)

# Access the best output for each task
best_per_task = result.detailed_results.best_outputs_valset
```

## What GEPA does NOT optimize

GEPA only tunes the **instruction string** (the Signature docstring). Everything else in your prompt is fixed during optimization:

| Prompt element | Optimized by GEPA? | Where it lives |
|----------------|-------------------|----------------|
| Signature docstring | Yes | `"""Classify the text."""` |
| `InputField(desc=...)` | No | `dspy.InputField(desc="...")` |
| `OutputField(desc=...)` | No | `dspy.OutputField(desc="...")` |
| Pydantic `Field(description=...)` | No | `pydantic.Field(description="...")` |
| Field names | No | `label: str = dspy.OutputField()` |
| Type constraints | No | `Literal["a", "b"]`, Pydantic models |
| Few-shot demos | No (by design) | Added by other optimizers |

This matters most for **structured output tasks** where Pydantic field descriptions carry significant guidance for the LM. If your output schema has `Field(description="Invoice date in YYYY-MM-DD format")`, GEPA will never touch that description -- even if it's the source of failures.

### Workaround: flatten field descriptions into the instruction

To bring field descriptions into GEPA's optimization surface, serialize them into the instruction before optimizing, then extract back out:

```python
import dspy
import json
from pydantic import BaseModel, Field

# 1. Your original Pydantic model
class Invoice(BaseModel):
    vendor: str = Field(description="Company name of the vendor")
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    total: float = Field(description="Total amount due")

# 2. Serialize field descriptions into the instruction
field_guidance = "\n".join(
    f"- {name}: {info.description}"
    for name, info in Invoice.model_fields.items()
    if info.description
)

class ParseInvoice(dspy.Signature):
    # GEPA will optimize this entire docstring, including the field guidance
    f"""Extract invoice data from raw text.

    Output field guidelines:
    {field_guidance}"""

    text: str = dspy.InputField()
    invoice: Invoice = dspy.OutputField()

# 3. Optimize -- GEPA now sees and can rewrite the field guidance
gepa = dspy.GEPA(metric=metric, reflection_lm=reflection_lm, auto="medium")
optimized = gepa.compile(ParseInvoice, trainset=trainset)

# 4. After optimization, inspect the optimized instruction
# to see how GEPA refined the field guidance
dspy.inspect_history(n=1)
```

**Limitations of this workaround:**

- The optimized field guidance lives in the instruction string, not back in the Pydantic model. The Pydantic model still validates types, but its `description` fields remain unchanged.
- You must manually inspect the optimized instruction to see what GEPA changed about the field descriptions.
- For simple schemas (2-3 fields), this adds complexity with little benefit -- GEPA can usually compensate through the instruction alone.

**When this is worth doing:**

- Complex Pydantic models with 5+ fields where field descriptions carry important formatting or semantic guidance
- Structured output tasks where the LM consistently misinterprets specific fields despite good top-level instructions
- When field-level `desc` strings are doing heavy lifting (e.g., date formats, enum explanations, nested object guidance)

## Gotchas

1. **Claude writes GEPA metrics that return only a float.** GEPA can use plain float scores, but its key advantage is textual feedback. When the metric returns `{"score": 0.0, "feedback": "Expected positive but got negative; the review is sarcastic"}`, the reflection LM uses that feedback to propose better instructions. Without feedback, GEPA degrades to blind search. Always return a dict with both `score` and `feedback`.
2. **Claude uses a weak model as the reflection LM.** The quality of proposed instructions depends entirely on the reflection model. Using `gpt-4o-mini` or a small local model for reflection produces generic, unhelpful instruction changes. Use a strong model (GPT-4o, Claude Sonnet) for `reflection_lm` -- the task LM can be cheaper.
3. **Claude starts with `auto="heavy"` before validating the metric.** A broken or noisy metric wastes the entire optimization budget. Start with `auto="light"` to verify the metric produces meaningful scores and feedback, then scale up to `"medium"` or `"heavy"` for production runs.
4. **Claude does not run `dspy.Evaluate` before and after GEPA.** Without a baseline measurement, there is no way to know if GEPA actually improved anything. Always evaluate the unoptimized program first, then compare against the optimized version.
5. **Claude expects GEPA to optimize Pydantic field descriptions.** GEPA only tunes the signature docstring (instruction). `InputField(desc=...)`, `OutputField(desc=...)`, and Pydantic `Field(description=...)` are never modified. If field descriptions are causing failures, flatten them into the instruction before optimizing (see the workaround in this skill).

## Additional resources

- [dspy.GEPA API docs](https://dspy.ai/api/optimizers/GEPA/)
- [DSPy optimizer selection guide](https://dspy.ai/learn/optimization/optimizers/)
- For constructor signatures and method reference, see [reference.md](reference.md)
- For worked examples (sentiment classification, multi-step summarization), see [examples.md](examples.md)

## Cross-references

> Install any skill: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill <name>`

- **Improving accuracy** with other optimizers -- see `/ai-improving-accuracy`
- **MIPROv2** for instruction + few-shot optimization -- see `/dspy-miprov2`
- **Chain of thought reasoning** as the inner module -- see `/dspy-chain-of-thought`
- **Evaluating programs** before and after optimization -- see `/dspy-evaluate`
- **Iterative self-improvement** at inference time -- see `/dspy-refine`
- **Signatures and Pydantic outputs** -- see `/dspy-signatures` for field descriptions, typed outputs, and gotchas about what optimizers can/cannot tune
- **VizPy** (commercial alternative for instruction optimization) -- see `/dspy-vizpy`
- For worked examples, see [examples.md](examples.md)
- **Install `/ai-do` if you do not have it** — it routes any AI problem to the right skill and is the fastest way to work: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill ai-do`
