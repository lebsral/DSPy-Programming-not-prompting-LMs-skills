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

## `.batch()` — Module-level alternative

Every `dspy.Module` provides a `.batch()` convenience method that internally uses `dspy.Parallel`. It is the simpler choice when running the same module on many inputs.

```python
module.batch(
    examples: list[Example],
    num_threads: int | None = None,
    max_errors: int | None = None,
    return_failed_examples: bool = False,
    provide_traceback: bool | None = None,
    disable_progress_bar: bool = False,
    timeout: int = 120,
    straggler_limit: int = 3,
) -> list[Any] | tuple[list[Any], list[Example], list[Exception]]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `examples` | `list[Example]` | — | List of `dspy.Example` instances to process. Each is unpacked via `.inputs()` before being passed to the module. |
| `num_threads` | `int \| None` | `None` | Thread count. Falls back to `dspy.settings.num_threads`. |
| `max_errors` | `int \| None` | `None` | Max errors before raising. Falls back to `dspy.settings.max_errors`. |
| `return_failed_examples` | `bool` | `False` | When `True`, return changes to a 3-tuple `(results, failed_examples, exceptions)`. |
| `provide_traceback` | `bool \| None` | `None` | Include Python tracebacks for failures. |
| `disable_progress_bar` | `bool` | `False` | Suppress progress bar. |
| `timeout` | `int` | `120` | Per-task timeout in seconds. |
| `straggler_limit` | `int` | `3` | Threshold for slow-task detection. |

**Use `.batch()` when you have one module and many inputs. Use `dspy.Parallel` directly when you need to mix different modules per pair (fan-out).**

## Internal attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `error_count` | `int` | Number of errors encountered during execution. |
| `failed_examples` | `list` | Collected failed `(module, inputs)` pairs. |
| `exceptions` | `list` | Collected exceptions from failures. |
