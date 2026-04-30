# RLM API Reference

> Condensed from [dspy.ai/api/modules/RLM](https://dspy.ai/api/modules/RLM). Verify against upstream for latest.

## Constructor

```python
dspy.RLM(
    signature,                # str | Signature -- required
    max_iterations=20,        # max REPL interaction loops
    max_llm_calls=50,         # max sub-LM queries per execution
    max_output_chars=10_000,  # max chars from REPL output per step
    verbose=False,            # detailed execution logging
    tools=None,               # list[Callable] -- custom tool functions
    sub_lm=None,              # dspy.LM -- cheaper LM for sub-queries
    interpreter=None,         # CodeInterpreter (defaults to PythonInterpreter)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Defines inputs/outputs |
| `max_iterations` | `int` | `20` | Max REPL interaction cycles |
| `max_llm_calls` | `int` | `50` | Max `llm_query()` calls per run |
| `max_output_chars` | `int` | `10_000` | Char limit for REPL output per step |
| `verbose` | `bool` | `False` | Enable detailed logging |
| `tools` | `list[Callable] \| None` | `None` | Custom callable tools for the sandbox |
| `sub_lm` | `dspy.LM \| None` | `None` | Separate LM for `llm_query()` calls |
| `interpreter` | `CodeInterpreter \| None` | `None` | Custom code execution environment |

## Key Methods

- `forward(**inputs) -> dspy.Prediction` -- run RLM with provided inputs
- `aforward(**inputs) -> dspy.Prediction` -- async variant
- `batch(examples, num_threads, max_errors, ...)` -- parallel processing

## Built-in REPL Functions

| Function | Description |
|----------|-------------|
| `llm_query(prompt)` | Query the sub-LM (up to ~500K chars) |
| `llm_query_batched(prompts)` | Concurrent multi-prompt queries |
| `print()` | Display output (required to see results) |
| `SUBMIT(output)` | End execution and return final answer |
