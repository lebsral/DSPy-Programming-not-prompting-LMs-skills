> Condensed from [dspy.ai/api/modules/Module/](https://dspy.ai/api/modules/Module/). Verify against upstream for latest.

# dspy.Module — API Reference

## Constructor

```python
dspy.Module(callbacks=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callbacks` | `list \| None` | `None` | Optional callback handlers for instrumentation and monitoring |

Subclass `dspy.Module` and implement `forward()` to define your program logic.

## Key methods

### Execution

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(*args, **kwargs) -> Prediction` | Invokes `forward()` with callback support and usage tracking |
| `acall` | `async (*args, **kwargs) -> Prediction` | Async version of `__call__` |
| `forward` | `(*args, **kwargs)` | Must be implemented by subclasses to define program logic |
| `batch` | `(examples, num_threads=None, max_errors=None, return_failed_examples=False, timeout=120) -> list` | Process multiple examples in parallel |

### State management

| Method | Signature | Description |
|--------|-----------|-------------|
| `save` | `(path, save_program=False, modules_to_serialize=None)` | Save module state to JSON. `save_program=True` saves full program to directory |
| `load` | `(path, allow_pickle=False, allow_unsafe_lm_state=False)` | Load saved module state |
| `dump_state` | `(json_mode=True) -> dict` | Export current state as dictionary |
| `load_state` | `(state, allow_unsafe_lm_state=False)` | Restore state from dictionary |

### Introspection

| Method | Signature | Description |
|--------|-----------|-------------|
| `named_predictors` | `() -> list[tuple[str, Predict]]` | Returns all (name, Predict) pairs in the module tree |
| `predictors` | `() -> list[Predict]` | Returns all Predict instances |
| `named_sub_modules` | `(type_=None, skip_compiled=False) -> Generator` | Finds all sub-modules with their paths |
| `named_parameters` | `() -> list` | Returns all parameters including those in nested lists/dicts |
| `inspect_history` | `(n=1, file=None) -> None` | Print last n LM interactions |

### Language model management

| Method | Signature | Description |
|--------|-----------|-------------|
| `set_lm` | `(lm) -> None` | Sets language model for all predictors recursively |
| `get_lm` | `() -> LM` | Returns the LM if all predictors share one; raises ValueError if multiple |

### Utilities

| Method | Signature | Description |
|--------|-----------|-------------|
| `map_named_predictors` | `(func) -> self` | Apply function to all Predict instances, returns self for chaining |
| `deepcopy` | `() -> Module` | Deep copy prioritizing parameter copying |
| `reset_copy` | `() -> Module` | Deep copy with all parameters reset |

## batch() details

```python
module.batch(
    examples,                      # list[dspy.Example]
    num_threads=None,              # parallel threads (default: dspy.settings.num_threads)
    max_errors=None,               # max errors before stopping
    return_failed_examples=False,  # if True, returns (results, failed, exceptions)
    timeout=120,                   # per-example timeout in seconds
    straggler_limit=3,             # max straggler threads to wait for
)
```

## save() / load() details

```python
# Save optimized state (few-shot demos, instructions)
optimized.save("my_program.json")

# Save full program including code (experimental)
optimized.save("my_program_dir/", save_program=True)

# Load into a fresh instance
program = MyProgram()
program.load("my_program.json")
```

What gets saved: few-shot demonstrations, optimized instructions, Predict module state.
What does NOT get saved: Python logic in `forward()`, model weights, LM configuration.
