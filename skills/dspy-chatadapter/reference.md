> Condensed from [dspy.ai/api/adapters/ChatAdapter/](https://dspy.ai/api/adapters/ChatAdapter/). Verify against upstream for latest.

# dspy.ChatAdapter — API Reference

## Constructor

```python
dspy.ChatAdapter(
    callbacks=None,                    # list[BaseCallback] | None
    use_native_function_calling=False, # bool
    native_response_types=None,        # list[type] | None
    use_json_adapter_fallback=True,    # bool
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callbacks` | `list[BaseCallback] | None` | `None` | Callback hooks executed during format/parse operations. |
| `use_native_function_calling` | `bool` | `False` | Use provider-native function calling for structured output (OpenAI, Anthropic). |
| `native_response_types` | `list[type] | None` | `None` | Output field types handled by native LM features instead of text parsing. |
| `use_json_adapter_fallback` | `bool` | `True` | Automatically retry with JSONAdapter when ChatAdapter parsing fails. |

## Methods

### Core call methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `__call__` | `(lm, lm_kwargs, signature, demos, inputs)` | `list[dict[str, Any]]` | Execute adapter with automatic JSON fallback on errors (except context window exceeded). |
| `acall` | `(lm, lm_kwargs, signature, demos, inputs)` | `list[dict[str, Any]]` | Async version with identical fallback behavior. |

### Format methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `format` | `(signature, demos, inputs)` | `list[dict[str, Any]]` | Convert signature + demos + inputs into multiturn chat messages (system, user, assistant). |
| `format_system_message` | `(signature)` | `str` | Generate system instructions from signature docstring, field descriptions, and type constraints. |
| `format_user_message_content` | `(signature, inputs, prefix='', suffix='', main_request=False)` | `str` | Structure user input with `[[ ## field ## ]]` markers and optional output format reminders. |
| `user_message_output_requirements` | `(signature)` | `str` | Return lightweight format reminder for long conversations to maintain output structure awareness. |

### Parse methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `parse` | `(signature, completion)` | `dict[str, Any]` | Extract output fields from LM response using `[[ ## field_name ## ]]` delimiters. Raises `AdapterParseError` on failure. |

### Fine-tuning

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `format_finetune_data` | `(signature, demos, inputs, outputs)` | `dict[str, list[Any]]` | Format data for OpenAI API fine-tuning as `{"messages": [system, user, assistant]}`. |

## Field delimiter protocol

ChatAdapter uses `[[ ## field_name ## ]]` headers to delimit fields in both prompts and responses:

```
[[ ## question ## ]]
What is the capital of France?

[[ ## answer ## ]]
Paris

[[ ## completed ## ]]
```

The `[[ ## completed ## ]]` marker signals that all output fields have been provided.

## Fallback behavior

```
ChatAdapter.parse() succeeds? → Return result
                     fails?   → ContextWindowExceededError? → Re-raise
                               → Other error? → Retry entire call with JSONAdapter
```

## Per-module adapter assignment

```python
# Assign a different adapter to a specific module
my_module.set_adapter(dspy.JSONAdapter())
```

This overrides the global adapter for that module only.
