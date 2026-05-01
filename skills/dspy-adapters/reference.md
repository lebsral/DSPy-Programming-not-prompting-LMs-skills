> Condensed from [dspy.ai/api/adapters/](https://dspy.ai/api/adapters/Adapter/), [ChatAdapter/](https://dspy.ai/api/adapters/ChatAdapter/), [JSONAdapter/](https://dspy.ai/api/adapters/JSONAdapter/), [XMLAdapter/](https://dspy.ai/api/adapters/XMLAdapter/), and [TwoStepAdapter/](https://dspy.ai/api/adapters/TwoStepAdapter/). Verify against upstream for latest.

# DSPy Adapters — API Reference

## dspy.Adapter (base class)

```python
dspy.Adapter(
    callbacks=None,                    # list[BaseCallback] | None
    use_native_function_calling=False, # bool
    native_response_types=None,        # list[type] | None — defaults to [Citations]
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callbacks` | `list[BaseCallback] | None` | `None` | Callback functions for logging/monitoring during formatting and parsing. |
| `use_native_function_calling` | `bool` | `False` | Enable native LM function calling when inputs contain `dspy.Tool` types. |
| `native_response_types` | `list[type] | None` | `None` (defaults to `[Citations]`) | Output field types handled by native LM features (bypasses adapter formatting for those types). |

**Abstract class.** Subclass and implement `format_field_description`, `format_field_structure`, `format_task_description`, `parse`, `format_user_message_content`, and `format_assistant_message_content`.

### Key methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(lm, lm_kwargs, signature, demos, inputs) -> list[dict]` | Format → LM call → parse pipeline. |
| `acall` | async variant of `__call__` | Async pipeline. |
| `format` | `(signature, demos, inputs) -> list[dict]` | Convert inputs + demos into multiturn LM messages. |
| `parse` | `(signature, completion) -> dict` | Extract output fields from raw LM response (abstract). |
| `format_system_message` | `(signature) -> str` | Combine field descriptions, structure, and task instructions. |
| `format_demos` | `(signature, demos) -> list[dict]` | Transform few-shot examples into user/assistant message pairs. |
| `format_conversation_history` | `(signature, ...) -> list[dict]` | Format historical messages from History field. |

---

## dspy.ChatAdapter

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
| `callbacks` | `list[BaseCallback] | None` | `None` | Callback functions executed during adapter operations. |
| `use_native_function_calling` | `bool` | `False` | Enable native function calling features of the LM provider. |
| `native_response_types` | `list[type] | None` | `None` | Output field types handled natively by the LM (bypasses adapter formatting for those types). |
| `use_json_adapter_fallback` | `bool` | `True` | Automatically retry with JSONAdapter when parsing fails (except on context window exceeded). |

**Inheritance:** Extends `Adapter`.

### Key methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(lm, lm_kwargs, signature, demos, inputs) -> list[dict]` | Execute adapter with fallback logic. |
| `acall` | async variant of `__call__` | Async execution with fallback. |
| `format` | `(signature, demos, inputs) -> list[dict]` | Convert DSPy signature + inputs into chat messages. |
| `parse` | `(signature, completion) -> dict` | Extract output fields from LM response text. |
| `format_system_message` | `(signature) -> str` | Generate the system message. |
| `format_demos` | `(signature, demos) -> list[dict]` | Format few-shot examples as user/assistant message pairs. |
| `format_finetune_data` | `(signature, demos, inputs, outputs) -> dict` | Prepare data in OpenAI fine-tuning format. |

---

## dspy.JSONAdapter

```python
dspy.JSONAdapter(
    callbacks=None,                    # list[BaseCallback] | None
    use_native_function_calling=True,  # bool (note: True by default, unlike ChatAdapter)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callbacks` | `list[BaseCallback] | None` | `None` | Callback functions executed during adapter operations. |
| `use_native_function_calling` | `bool` | `True` | Use native function calling (enabled by default). |

**Inheritance:** Extends `ChatAdapter`.

### Key methods

Inherits all ChatAdapter methods. Overrides:

| Method | Description |
|--------|-------------|
| `__call__` | Attempts structured output format first, falls back to JSON mode on failure. |
| `parse` | Extracts and validates JSON from LM responses using `json_repair`. |
| `format_field_structure` | Generates JSON schema instructions instead of field delimiters. |
| `format_user_message_content` | Adds JSON formatting instructions to user messages. |
| `format_assistant_message_content` | Formats assistant responses as JSON. |

### Key behavior

- Instructs the LM to respond with a JSON object matching output field names and types.
- Uses provider native structured output mode when available, falls back to `response_format: {"type": "json_object"}`.
- Parses with `json_repair` for resilience against minor formatting errors.
- Raises `AdapterParseError` if response cannot be parsed.

---

## dspy.XMLAdapter

```python
dspy.XMLAdapter(
    callbacks=None,  # list[BaseCallback] | None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callbacks` | `list[BaseCallback] | None` | `None` | Callback functions executed during adapter operations. |

**Inheritance:** Extends `ChatAdapter`.

### Key methods

Inherits all ChatAdapter methods. Overrides:

| Method | Description |
|--------|-------------|
| `format_field_structure` | Generates XML tag instructions instead of field delimiters. |
| `format_user_message_content` | Wraps input fields in XML tags. |
| `format_assistant_message_content` | Formats assistant responses with XML tags. |
| `parse` | Extracts field values from XML tags using regex pattern matching. |
| `user_message_output_requirements` | Specifies XML tag format requirements. |

### Key behavior

- Wraps fields in XML tags: `<field_name>value</field_name>`.
- Falls back to `JSONAdapter` on parse failure (like ChatAdapter).
- Simpler constructor than ChatAdapter — only takes `callbacks`.

---

## dspy.TwoStepAdapter

```python
dspy.TwoStepAdapter(
    extraction_model,    # dspy.BaseLM (required)
    **kwargs,            # passed to parent Adapter
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extraction_model` | `dspy.BaseLM` | required | Smaller LM for extracting structured data from the main LM response. Must be a `BaseLM` instance. |
| `**kwargs` | | | Additional arguments passed to the parent `Adapter` class. |

**Inheritance:** Extends `Adapter`.

**Raises:** `ValueError` if `extraction_model` is not a `BaseLM` instance.

### Key methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(lm, lm_kwargs, signature, demos, inputs) -> list[dict]` | Two-stage pipeline: format → call main LM → extract with extraction_model. |
| `acall` | async variant | Async two-stage pipeline. |
| `format` | `(signature, demos, inputs) -> list[dict]` | Formats natural language prompt (no formatting constraints) for the main LM. |
| `parse` | `(signature, completion) -> dict` | Uses extraction_model with ChatAdapter to structure the main LM raw text. |

### Key behavior

- Main LM receives a simplified, natural language prompt — no field delimiters or JSON instructions.
- Extraction model uses ChatAdapter internally to parse freeform response into structured fields.
- Preserves `tool_calls` and `logprobs` from main LM responses.
- Two LM calls per prediction (main + extraction).

---

## Configuration methods

| Method | Where | Description |
|--------|-------|-------------|
| `dspy.configure(adapter=...)` | Global | Set adapter for all modules. |
| `dspy.context(adapter=...)` | Block | Temporary override within a `with` block. |
| `module.set_adapter(adapter)` | Per-module | Override adapter for a specific module only. |
