---
name: dspy-bootstrap-finetune
description: "Use when you need maximum quality from a smaller/cheaper model — generates training data from a teacher model and fine-tunes a student model's weights. Common scenarios: distilling GPT-4 quality into a cheaper model, generating training data from a strong teacher to fine-tune a weak student, reducing inference costs by replacing an expensive model with a fine-tuned small one, or building a production model that's fast and cheap. Related: ai-fine-tuning, ai-cutting-costs, dspy-better-together. Also: "dspy.BootstrapFinetune", "model distillation with DSPy", "teacher-student training", "fine-tune small model from GPT-4 outputs", "reduce API costs with fine-tuning", "generate training data then fine-tune", "cheap model same quality", "distill large model into small model", "fine-tune Llama from GPT-4", "production model training", "move from API to self-hosted model"."
---

# Fine-Tune LM Weights with dspy.BootstrapFinetune

Guide the user through using DSPy's `BootstrapFinetune` optimizer to automatically generate training data from successful reasoning traces and fine-tune a language model's weights. This is the heaviest optimization DSPy offers -- it changes the model itself, not just the prompt.

## What is BootstrapFinetune

`dspy.BootstrapFinetune` is an optimizer that tunes LM **weights** rather than prompts. It works in two phases:

1. **Bootstrap**: Run your program on every training example, keep the traces where your metric passes.
2. **Fine-tune**: Send those successful traces to the model provider's fine-tuning API (or a local training loop) and train the model weights on them.

The result is a version of your program backed by a fine-tuned model that has internalized the reasoning patterns from the bootstrapped traces.

```
Training examples ──> Run program ──> Keep passing traces ──> Fine-tune model weights
```

## When to use BootstrapFinetune

Use it when:

- You have **500+ labeled examples** (1000+ is better -- more data means more successful traces to train on)
- You have already tried **prompt optimization** (MIPROv2, BootstrapFewShot) and hit a quality ceiling
- You want a **smaller, cheaper model** to match the quality of a larger one (model distillation)
- You need **maximum quality** and are willing to pay the one-time cost of fine-tuning
- Your domain has **specialized patterns** that the base model doesn't handle well out of the box

Do **not** use it when:

- You have fewer than 500 examples -- use `/ai-improving-accuracy` with MIPROv2 or BootstrapFewShot instead
- You haven't tried prompt optimization yet -- start there, it's 10x cheaper
- Your baseline accuracy is below 50% -- fix your task definition or data first
- You're still iterating on what the task is -- fine-tuning locks you into a specific behavior
- You don't have a clear, automated metric -- you can't filter traces without one

## Basic usage

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 1. Define your program
class Classify(dspy.Signature):
    """Classify the support ticket category."""
    text: str = dspy.InputField()
    category: str = dspy.OutputField()

program = dspy.ChainOfThought(Classify)

# 2. Prepare labeled data (500+ examples)
trainset = [
    dspy.Example(text="Can't log in", category="auth").with_inputs("text"),
    dspy.Example(text="Charge me twice", category="billing").with_inputs("text"),
    # ... 500+ examples
]

# 3. Define a metric
def metric(example, prediction, trace=None):
    return prediction.category.lower() == example.category.lower()

# 4. Fine-tune
optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=24)
finetuned = optimizer.compile(program, trainset=trainset)

# 5. Use the fine-tuned program
result = finetuned(text="My payment failed")
print(result.category)
```

After `compile` finishes, `finetuned` is a copy of your program that uses the newly fine-tuned model. Every module in the program that was backed by a fine-tunable LM gets updated.

## Teacher-student paradigm

The most powerful pattern: use an expensive, high-quality model (the teacher) to generate traces, then fine-tune a cheap model (the student) on those traces. This is model distillation.

```python
# --- Teacher: expensive model, high quality ---
teacher_lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=teacher_lm)

teacher = dspy.ChainOfThought(Classify)

# Optionally optimize the teacher's prompts first for even better traces
prompt_optimizer = dspy.MIPROv2(metric=metric, auto="medium")
teacher_optimized = prompt_optimizer.compile(teacher, trainset=trainset)

# --- Student: cheap model, fine-tuned on teacher's traces ---
student_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=student_lm)

student = dspy.ChainOfThought(Classify)

ft_optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=24)
student_finetuned = ft_optimizer.compile(
    student,
    trainset=trainset,
    teacher=teacher_optimized,  # Teacher generates the traces
)
```

How it works with a teacher:

1. The **teacher** program runs on each training example using the expensive model
2. Only traces where the metric passes are kept
3. Those traces are reformatted as training data for the **student** model
4. The student model is fine-tuned on the teacher's successful reasoning patterns

The student learns to mimic the teacher's reasoning at a fraction of the inference cost.

## Target model configuration

BootstrapFinetune fine-tunes whatever LM is configured when you call `compile`. To control which model gets fine-tuned:

```python
# Fine-tune GPT-4o-mini
student_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=student_lm)
finetuned = optimizer.compile(student, trainset=trainset)

# Fine-tune an open-source model via Together AI
student_lm = dspy.LM("together_ai/meta-llama/Llama-3-70b-chat-hf")
dspy.configure(lm=student_lm)
finetuned = optimizer.compile(student, trainset=trainset)
```

The model must support fine-tuning through its provider's API. Common options:

| Provider | Fine-tunable models | Notes |
|----------|-------------------|-------|
| OpenAI | `gpt-4o-mini`, `gpt-4o` | Easiest setup, DSPy handles the API calls |
| Together AI | Llama, Mistral, etc. | Open-source models, competitive pricing |
| Local | Any HuggingFace model | Full control, needs GPU(s) |

## Key parameters

```python
dspy.BootstrapFinetune(
    metric,          # Scoring function: (example, prediction, trace) -> bool/float
    num_threads=24,  # Parallel threads for bootstrapping traces
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `metric` | `Callable` | Scores each trace during bootstrapping. Only passing traces become training data. Same signature as evaluation metrics: `(example, prediction, trace=None) -> bool or float` |
| `num_threads` | `int` | Number of parallel threads for running the program on training examples. Higher = faster bootstrapping but more concurrent API calls. Default varies; 24 is a good starting point. |

The `compile` method accepts:

```python
optimizer.compile(
    program,         # Your dspy.Module to fine-tune
    trainset,        # List of dspy.Example with labeled data
    teacher=None,    # Optional: a teacher program for distillation
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `program` | `dspy.Module` | The program whose backing LM will be fine-tuned |
| `trainset` | `list[dspy.Example]` | Labeled training data (500+ recommended) |
| `teacher` | `dspy.Module or None` | If provided, the teacher generates traces instead of the student. Use for distillation. |

## Computational cost

BootstrapFinetune is the most expensive optimizer in DSPy. Budget for three cost stages:

### 1. Bootstrapping (LM API calls)

Every training example gets run through your program (or the teacher). With 1000 examples and a ChainOfThought module, that's 1000+ LM calls just for bootstrapping.

- **With teacher (GPT-4o)**: ~$5-15 for 1000 examples (depends on input/output length)
- **Without teacher (GPT-4o-mini)**: ~$0.15-0.50 for 1000 examples

### 2. Fine-tuning (provider charges)

The model provider charges for training. Costs depend on the number of successful traces and their length.

- **OpenAI GPT-4o-mini**: ~$0.008/1K training tokens
- **OpenAI GPT-4o**: ~$0.025/1K training tokens
- **Together AI**: Varies by model, generally cheaper for open-source

### 3. Inference (ongoing)

Fine-tuned models may cost slightly more per token than base models (OpenAI charges ~1.5x for fine-tuned inference). But if you distilled from GPT-4o to GPT-4o-mini, the net savings are still 10-30x.

### Time

- Bootstrapping: minutes to an hour (depends on dataset size and thread count)
- Fine-tuning: 30 minutes to several hours (depends on provider and dataset size)
- Total: plan for 1-4 hours end to end

## When to use BootstrapFinetune vs prompt optimization

| Factor | Prompt optimization (MIPROv2) | BootstrapFinetune |
|--------|------------------------------|-------------------|
| **What it changes** | Prompt instructions + few-shot examples | Model weights |
| **Data needed** | ~200 examples | ~500+ examples |
| **Cost** | Low (just LM calls for optimization) | High (LM calls + fine-tuning fees) |
| **Time** | Minutes | Hours |
| **Quality ceiling** | Good, but limited by what prompts can do | Higher -- model learns domain patterns |
| **Portability** | Optimized prompts work with any model | Weights are locked to one model |
| **Iteration speed** | Fast -- re-optimize in minutes | Slow -- re-train takes hours |
| **Best for** | Early development, quick iteration | Production, maximum quality, cost reduction via distillation |

**Recommended progression:**

1. Start with `dspy.BootstrapFewShot` (quick, ~50 examples)
2. Graduate to `dspy.MIPROv2` (better, ~200 examples)
3. Use `dspy.BootstrapFinetune` when prompt optimization plateaus (500+ examples)
4. Try `dspy.BetterTogether` for absolute maximum quality (combines prompt + weight optimization)

## Save and load

```python
# Save the fine-tuned program
finetuned.save("finetuned_classify.json")

# Load later for production
from my_module import MyProgram
production = MyProgram()
production.load("finetuned_classify.json")
result = production(text="New ticket text...")
```

The saved file stores the fine-tuned model identifier (e.g., `ft:gpt-4o-mini-2024-07-18:org::abc123`) so loading automatically points to the right model.

## Troubleshooting

### Not enough successful traces

If only a small fraction of training examples produce passing traces, the fine-tuning data will be thin.

**Fixes:**
- Use a stronger teacher model (GPT-4o instead of GPT-4o-mini)
- Relax your metric temporarily (accept partial credit during bootstrapping)
- Simplify your task or break multi-step programs into single steps
- Add more training examples so even a low success rate yields enough traces

### Overfitting (high train accuracy, low test accuracy)

**Fixes:**
- Add more training data
- Reduce fine-tuning epochs (if your provider exposes this setting)
- Use a larger base model (less prone to memorization)
- Simplify output format

### Fine-tuning didn't beat prompt optimization

**Fixes:**
- Verify bootstrapping produced 200+ successful traces (check logs)
- Try `dspy.BetterTogether` to combine prompt and weight optimization
- Confirm your metric correlates with actual quality
- Try a different base model

## Cross-references

- **BootstrapFewShot** for lighter optimization without fine-tuning -- see `/ai-improving-accuracy`
- **Fine-tuning workflow** for the full decision framework, prerequisites, and BetterTogether -- see `/ai-fine-tuning`
- **Cost reduction** for distillation and other strategies to cut API spend -- see `/ai-cutting-costs`
- For worked examples (distillation, production cost reduction), see [examples.md](examples.md)
