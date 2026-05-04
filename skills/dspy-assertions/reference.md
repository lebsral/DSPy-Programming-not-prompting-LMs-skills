> Condensed from [dspy.ai/learn/programming/7-assertions/](https://dspy.ai/learn/programming/7-assertions/) and [dspy/primitives/assertions.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/primitives/assertions.py). Verify against upstream for latest.

# dspy.Assert and dspy.Suggest — API Reference

> **REMOVED IN DSPy 3.x.** `dspy.Assert` and `dspy.Suggest` have been removed from the DSPy codebase — no `assertions.py` file, no imports in any `__init__.py`, `retry.py` is entirely commented out, and the docs page is gone. This reference is kept for maintaining legacy codebases only. **For new code, use `dspy.Refine` or `dspy.BestOfN`** — see `/dspy-refine` and `/dspy-best-of-n`.

## dspy.Assert

```python
dspy.Assert(
    condition,               # bool (required)
    message="",              # str
    backtrack_module=None,   # dspy.Module | None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `condition` | `bool` | required | The constraint to enforce. `False` triggers retry/backtrack. |
| `message` | `str` | `""` | Feedback message injected into the LM prompt on retry. Should be actionable — the LM reads this to fix its output. |
| `backtrack_module` | `dspy.Module | None` | `None` | Which module to backtrack to. If `None`, backtracks to the most recent LM call. |

**Behavior when `condition` is `False`:**

1. Raises `DSPyAssertionError` internally
2. DSPy catches it and injects `message` into the LM's context
3. Retries the target module (most recent call, or `backtrack_module` if specified)
4. Repeats up to `max_backtrack_attempts` times (default: 2)
5. If all retries fail, raises `DSPyAssertionError` to the caller

**During optimization:** Failed assertions cause the training example to be retried. If constraints cannot be satisfied after retries, the example is skipped.

## dspy.Suggest

```python
dspy.Suggest(
    condition,               # bool (required)
    message="",              # str
    backtrack_module=None,   # dspy.Module | None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `condition` | `bool` | required | The soft constraint to check. `False` logs a warning. |
| `message` | `str` | `""` | Warning message logged when condition fails. |
| `backtrack_module` | `dspy.Module | None` | `None` | Ignored for Suggest (no backtracking occurs). Included for API symmetry with Assert. |

**Behavior when `condition` is `False`:**

1. Raises `DSPySuggestionError` internally
2. DSPy catches it and logs a warning
3. Execution continues normally — no retry, no error raised to caller

**During optimization:** Suggestion failures are tracked as soft signals. Optimizers prefer demo sets where suggestions are satisfied, but violations do not prevent examples from being used.

## DSPyAssertionError

```python
from dspy.primitives.assertions import DSPyAssertionError
```

Raised when `dspy.Assert` exhausts all retry attempts. Catch this at the call site for graceful error handling:

```python
try:
    result = program(question="...")
except DSPyAssertionError as e:
    print(f"Validation failed: {e}")
```

## DSPySuggestionError

```python
from dspy.primitives.assertions import DSPySuggestionError
```

Raised internally by `dspy.Suggest` when the condition fails. DSPy catches this automatically — you do not normally need to handle it.

## Backtracking mechanics

- Assertions only work inside `dspy.Module.forward()` — DSPy needs the module context to manage retries
- On failure, DSPy modifies the signature by appending the error message as additional context
- The default `max_backtrack_attempts` is 2 (configurable via `dspy.settings`)
- Each retry uses a fresh LM call with the feedback message included
- If `backtrack_module` is specified, DSPy retries that specific module instead of the most recent one

## Key differences summary

| | `dspy.Assert` | `dspy.Suggest` |
|---|---|---|
| Severity | Hard — must pass | Soft — should pass |
| On failure | Retries with feedback, then raises error | Logs warning, continues |
| Exception type | `DSPyAssertionError` | `DSPySuggestionError` (caught internally) |
| Backtracking | Yes — retries the target module | No — execution continues |
| Optimization effect | Failed examples retried, then skipped | Soft signal for demo selection |
| Use for | Format, safety, schema constraints | Style, quality, preference nudges |
