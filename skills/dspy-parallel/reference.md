> Condensed from [dspy.ai/api/modules/Parallel/](https://dspy.ai/api/modules/Parallel/). Verify against upstream for latest.

# dspy.Parallel — API Reference

## Constructor

```python
dspy.Parallel(
    num_threads: int | None = None,
    max_errors: int | None = None,
    access_examples: bool = True,
    return_failed_examples: bool = False,
    provide_traceback: bool | None = None,
    disable_progress_bar: bool = False,
    timeout: int = 120,
    straggler_limit: int = 3,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_threads` | `int \| None` | `None` | Number of concurrent threads. Falls back to `dspy.settings.num_threads`. |
| `max_errors` | `int \| None` | `None` | Maximum errors before raising. Falls back to `dspy.settings.max_errors`. |
| `access_examples` | `bool` | `True` | Unpack `Example` objects via `.inputs()`. Set `False` to pass raw Examples to the module. |
| `return_failed_examples` | `bool` | `False` | When `True`, return value changes to a 3-tuple `(results, failed_examples, exceptions)`. |
| `provide_traceback` | `bool \| None` | `None` | Include Python tracebacks for failed examples in error output. |
| `disable_progress_bar` | `bool` | `False` | Suppress the tqdm progress bar. |
| `timeout` | `int` | `120` | Max seconds per individual task before timeout. |
| `straggler_limit` | `int` | `3` | Threshold for flagging slow-running tasks. |

## Methods

### `__call__(exec_pairs, num_threads=None)`

Delegates to `forward()`.

### `forward(exec_pairs, num_threads=None)`

Executes module-input pairs in parallel using a thread pool.

```python
results = parallel(exec_pairs)
# or with return_failed_examples=True:
results, failed, exceptions = parallel(exec_pairs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `exec_pairs` | `list[tuple[Module, dict \| Example]]` | List of `(module, inputs)` pairs to execute concurrently. |
| `num_threads` | `int \| None` | Override the instance-level thread count for this call. |

**Returns:**

- When `return_failed_examples=False` (default): `list[Any]` — results in same order as `exec_pairs`.
- When `return_failed_examples=True`: `tuple[list[Any], list[tuple], list[Exception]]` — `(results, failed_pairs, exceptions)`.

## Supported input formats

Each pair in `exec_pairs` is `(module, inputs)` where `inputs` can be:

| Format | Example |
|--------|---------|
| `dict` | `(module, {"question": "What is DSPy?"})` |
| `dspy.Example` | `(module, dspy.Example(question="...").with_inputs("question"))` |
| `list` | `(module, ["What is DSPy?"])` |
| `tuple` | `(module, ("What is DSPy?",))` |

## Internal attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `error_count` | `int` | Number of errors encountered during execution. |
| `failed_examples` | `list` | Collected failed `(module, inputs)` pairs. |
| `exceptions` | `list` | Collected exceptions from failures. |
