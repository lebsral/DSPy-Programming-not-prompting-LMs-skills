# Error Fixing Reference

## Error Index

### Setup Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: 'NoneType'` on LM calls | LM not configured | `dspy.configure(lm=dspy.LM("..."))` |
| `AuthenticationError` | Invalid API key | Check `OPENAI_API_KEY` or relevant env var |
| `ModuleNotFoundError: No module named 'dspy'` | DSPy not installed | `pip install -U dspy` |
| `ImportError: cannot import name 'X'` | Wrong DSPy version | `pip install -U dspy` to get latest |

### Signature Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `TypeError: forward() got an unexpected keyword argument` | Input field name mismatch | Match kwargs to InputField names |
| `ValueError: Could not parse output` | LM output format mismatch | Check `dspy.inspect_history()`, simplify types |
| `ValidationError` from Pydantic | Output doesn't match Pydantic model | Check type constraints, add field descriptions |
| Output field is `None` | LM didn't produce that field | Add clearer field descriptions, use ChainOfThought |

### Retriever Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Empty passages returned | Retriever not configured | `dspy.configure(rm=your_retriever)` |
| `ConnectionError` on retrieval | Retriever server down | Check server URL and connectivity |
| Wrong results retrieved | Bad query or wrong index | Test retriever directly, check index content |

### Optimization Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Score drops after optimization | Overfitting or bad metric | Use validation set, check metric manually |
| `ValueError` during compilation | Incompatible optimizer settings | Check optimizer requirements (data size, etc.) |
| Optimization takes too long | Too many trials or large data | Use `auto="light"`, reduce trainset size |
| `FileNotFoundError` on load | Wrong save path | Check the path passed to `.save()` |

### Runtime Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `RateLimitError` | Too many API calls | Add delays, reduce `num_threads`, use caching |
| `ContextLengthExceeded` | Prompt too long | Reduce `k` in retriever, reduce few-shot demos |
| Assertion/Suggestion failures | Output constraints not met | Make constraints less strict, improve descriptions |
| Infinite loop in ReAct | Agent can't find answer | Set `max_iters`, check tool implementations |

## Debugging Workflow

```
1. Does the AI respond?
   +- No -> Check API key, model name, network
   +- Yes v

2. Does the signature work standalone?
   +- No -> Fix signature (types, descriptions, docstring)
   +- Yes v

3. Does each module step work in isolation?
   +- No -> Fix the failing step
   +- Yes v

4. Does the full pipeline produce output?
   +- No -> Check data flow between steps
   +- Yes v

5. Is the output correct?
   +- No -> Check with dspy.inspect_history(), optimize
   +- Yes -> Done!
```

## Useful Debug Commands

```python
# See what LM is configured
print(dspy.settings.lm)

# See what retriever is configured
print(dspy.settings.rm)

# Inspect last N LM calls (prompts + responses)
dspy.inspect_history(n=3)

# Print module structure
print(my_program)

# List all predictors in a module
for name, pred in my_program.named_predictors():
    print(f"{name}: {type(pred).__name__}")

# Test LM directly
lm = dspy.LM("openai/gpt-4o-mini")
print(lm("Say hello"))

# Check prediction fields
result = my_program(question="test")
print(result.keys())  # available fields
print(result)          # all values
```

## Performance Troubleshooting

### Slow execution
- Reduce `num_threads` if hitting rate limits
- Use a faster/cheaper LM for development
- Cache LM calls: DSPy caches by default, but check if cache is being used

### High costs
- Use `gpt-4o-mini` or similar for development and optimization
- Reduce training set size during optimization iteration
- Use `MIPROv2(auto="light")` for quick optimization runs
- See `/ai-cutting-costs` for systematic cost reduction

### Inconsistent results
- Set `temperature=0` for deterministic outputs: `dspy.LM("...", temperature=0)`
- Run evaluation multiple times and average scores
- See `/ai-making-consistent` for systematic consistency improvement

## Observability

### Using DSPy's built-in tracing

```python
# Enable tracing
dspy.configure(lm=lm, trace=[])

# After running, inspect trace
result = my_program(question="test")
# trace contains the execution path
```

### Logging LM calls

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or selectively
logger = logging.getLogger("dspy")
logger.setLevel(logging.DEBUG)
```

### Counting LM calls and tokens

```python
# Check LM call history
print(f"Total LM calls: {lm.history}")
# Review token usage in the history entries
```
