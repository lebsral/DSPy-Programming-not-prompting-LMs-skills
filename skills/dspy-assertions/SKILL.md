---
name: dspy-assertions
description: Use when you need to enforce constraints on LM outputs with dspy.Assert and dspy.Suggest — hard failures that trigger retries, soft suggestions that log warnings, backtracking behavior, and integrating assertions with optimizers. Common scenarios: ensuring extracted emails are valid, forcing output to be valid JSON, rejecting answers that do not cite sources, retrying when classification confidence is too low, enforcing length limits on summaries, or catching hallucinated entities. Related: ai-following-rules, ai-checking-outputs, ai-stopping-hallucinations. Also: dspy.Assert, dspy.Suggest, runtime validation for LLM output, retry on bad output, backtracking on constraint violation, LLM keeps ignoring my rules, output validation with automatic retry, how to make LLM follow constraints, assertion-driven development for AI, guard rails in DSPy, fail fast on bad LLM output.
---

# Enforce Constraints with dspy.Assert and dspy.Suggest

Guide the user through adding runtime constraints to DSPy programs. Assertions let you declare what valid output looks like — DSPy handles retrying, backtracking, and feeding error messages back to the LM automatically.

## Two kinds of constraints

| | `dspy.Assert` | `dspy.Suggest` |
|---|---|---|
| **Severity** | Hard — must pass | Soft — should pass |
| **On failure** | Retries with feedback, then raises error | Logs a warning, continues execution |
| **Use for** | Format requirements, safety checks, non-negotiable rules | Style preferences, quality nudges, nice-to-haves |

```python
import dspy

class QA(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        result = self.answer(question=question)

        # Hard constraint — retries if violated
        dspy.Assert(
            len(result.answer) > 0,
            "Answer must not be empty",
        )

        # Soft constraint — logs warning but continues
        dspy.Suggest(
            len(result.answer.split()) >= 10,
            "Answer should be at least 10 words for completeness",
        )

        return result
```

## dspy.Assert(condition, message)

Call `dspy.Assert` with a boolean condition and a message. When the condition is `False`, DSPy:

1. Catches the failure
2. Appends your message to the LM's context as feedback
3. Retries the LM call that produced the failing output
4. Repeats up to `max_backtrack_attempts` times (default: 2)
5. If all retries fail, raises `DSPyAssertionError`

```python
dspy.Assert(
    result.answer != "I don't know",
    "You must provide a substantive answer based on the context",
)
```

**Write specific messages.** The message is injected back into the prompt on retry, so "Answer was 350 words, must be under 200" is far more useful than "too long."

## dspy.Suggest(condition, message)

Same signature as `Assert`, but non-blocking. When the condition is `False`:

1. The message is logged as a warning
2. Execution continues normally
3. During optimization, suggestions guide the optimizer toward better prompts

```python
dspy.Suggest(
    "however" not in result.answer.lower(),
    "Avoid hedging language like 'however' — be direct",
)
```

Use `Suggest` when the constraint improves quality but isn't a hard requirement.

## How backtracking works

When `dspy.Assert` fails inside a module's `forward()`, DSPy doesn't just retry the same call. It modifies the signature by injecting the error message as additional context, so the LM has feedback about what went wrong:

```
# Original prompt (simplified)
Question: What is DSPy?
Answer: [LM generates here]

# After assertion failure, retry prompt becomes:
Question: What is DSPy?
Previous attempt failed: "Answer was 350 words, must be under 200. Be concise."
Answer: [LM generates here with feedback]
```

This is why assertion messages should be **actionable instructions**, not just error descriptions.

### Targeting a specific module for backtracking

By default, DSPy backtracks to the most recent LM call. Use the `backtrack_module` parameter to target a specific module instead:

```python
dspy.Assert(
    is_valid_json(result.output),
    "Output must be valid JSON. Check for missing braces or trailing commas.",
    backtrack_module=self.generate,  # retry this specific module
)
```

## Common validation patterns

### Length constraints

```python
dspy.Assert(
    len(result.summary.split()) <= 50,
    f"Summary is {len(result.summary.split())} words, must be under 50",
)
```

### Format validation

```python
import re

dspy.Assert(
    re.match(r"^\d{4}-\d{2}-\d{2}$", result.date or ""),
    "Date must be in YYYY-MM-DD format",
)
```

### Content checks

```python
dspy.Assert(
    not any(phrase in result.answer.lower() for phrase in ["as an ai", "i cannot"]),
    "Do not include AI self-references in the answer",
)
```

### List output validation

```python
dspy.Assert(
    len(result.tags) >= 1,
    "Must assign at least one tag",
)
dspy.Assert(
    all(tag in VALID_TAGS for tag in result.tags),
    f"All tags must be from the valid set: {VALID_TAGS}",
)
```

### Grounding in sources

```python
# Check that the answer references at least one key term from the context
context_terms = set(word.lower() for p in context for word in p.split() if len(word) > 5)
answer_terms = set(word.lower() for word in result.answer.split())
overlap = context_terms & answer_terms
dspy.Assert(
    len(overlap) >= 3,
    "Answer must reference specific terms from the source passages",
)
```

## Using assertions with optimizers

Assertions work with all DSPy optimizers. During optimization:

- **`dspy.Assert`** failures cause the training example to be retried. If the program can't satisfy the constraint after retries, that example is skipped.
- **`dspy.Suggest`** failures are tracked as soft signals. Optimizers like `BootstrapFewShotWithRandomSearch` and `MIPROv2` prefer demo sets where suggestions are satisfied.

This means the optimizer learns prompts and demos that satisfy your constraints on the first try, reducing retries in production:

```python
program = QA()

optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=my_metric,
    max_bootstrapped_demos=4,
    num_candidate_programs=10,
)
optimized = optimizer.compile(program, trainset=trainset)
```

After optimization, the program will have few-shot demos that naturally produce outputs satisfying your assertions.

## Catching assertion errors

When all retries are exhausted, `dspy.Assert` raises `DSPyAssertionError`. Handle it at the call site:

```python
from dspy.primitives.assertions import DSPyAssertionError

try:
    result = program(question="...")
except DSPyAssertionError as e:
    # Log the failure, return a fallback, etc.
    print(f"Output failed validation: {e}")
```

## When to use Assert vs. Suggest

| Scenario | Use |
|----------|-----|
| Output must be valid JSON | `Assert` |
| Answer should be concise | `Suggest` |
| No PII in output | `Assert` |
| Prefer active voice | `Suggest` |
| Must cite sources | `Assert` |
| Avoid hedging language | `Suggest` |
| Output matches expected schema | `Assert` |
| Include a confidence score | `Suggest` |

**Rule of thumb:** If a bad output reaching users would be a bug, use `Assert`. If it would just be suboptimal, use `Suggest`.

## Cross-references

- **Problem-first framing** with worked examples — see `/ai-checking-outputs`
- **Stopping hallucinations** with grounding and citations — see `/ai-stopping-hallucinations`
- **Enforcing business rules** and content policies — see `/ai-following-rules`
- **Optimizers** that learn to satisfy constraints — see `/dspy-bootstrap-rs`, `/dspy-miprov2`
- Not sure which skill to use next? Try `/ai-do` to get routed to the right one
