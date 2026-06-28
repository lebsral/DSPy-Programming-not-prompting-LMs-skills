# DSPy Primitives API Reference

> Condensed from [dspy.ai/api/primitives/](https://dspy.ai/api/primitives/). Verify against upstream for latest.

## dspy.Image

### Constructor

```python
dspy.Image(url=None, *, download=False, verify=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `Any` | `None` | Image source: HTTP/HTTPS URL, GS URL, local file path, raw bytes, `PIL.Image.Image`, dict `{"url": value}`, or data URI |
| `download` | `bool` | `False` | Download remote URLs to infer MIME type |
| `verify` | `bool` | `True` | Verify SSL certificates when downloading |

### Key Methods

| Method | Description |
|--------|-------------|
| `format()` | Returns formatted image as list of dicts or string (cached) |
| `serialize_model()` | Serializes with custom type identifiers |

### Deprecated Class Methods

These still work but use the constructor instead:

| Method | Replacement |
|--------|-------------|
| `from_url(url, download=False)` | `dspy.Image(url=url, download=download)` |
| `from_file(file_path)` | `dspy.Image(url=file_path)` |
| `from_PIL(pil_image)` | `dspy.Image(url=pil_image)` |

## dspy.Audio

### Constructor

```python
dspy.Audio(data="<base64-string>", audio_format="wav")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `str` | required | Base64-encoded audio data |
| `audio_format` | `str` | required | Audio format (e.g., `"wav"`, `"mp3"`) |

### Class Methods (preferred for creating Audio objects)

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_file(file_path)` | `file_path: str -> Audio` | Read local audio file and encode as base64 |
| `from_url(url)` | `url: str -> Audio` | Download audio from URL and encode as base64 |
| `from_array(array, sampling_rate, format="wav")` | `array: Any, sampling_rate: int, format: str -> Audio` | Encode numpy array as base64 audio. Requires `soundfile` library |

### Key Methods

| Method | Description |
|--------|-------------|
| `format()` | Returns list of dicts with `type: "input_audio"` and base64 data |
| `validate_input(values)` | Accepts Audio instances or dicts with `data` and `audio_format` keys |

## dspy.Code

### Bracket Notation

```python
dspy.Code["python"]    # Python code type
dspy.Code["java"]      # Java code type
dspy.Code["sql"]       # SQL code type
# Any language string works
```

The bracket notation creates a parameterized type. DSPy formats it as a fenced markdown code block with the language tag.

### Input Coercion

`dspy.Code` accepts multiple input forms via `validate_input`:

| Input type | Behavior |
|------------|----------|
| `str` | Treated as raw code string |
| `dict` with `"code"` key | Extracts the code string |
| `Code` instance | Returned as-is |

### Key Methods

| Method | Description |
|--------|-------------|
| `description()` | Returns description indicating markdown code block format with language |
| `format()` | Returns formatted code string |
| `parse_lm_response(response)` | Parses LM response into Code object |

## dspy.History

### Constructor

```python
dspy.History(messages=[...])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[dict[str, Any]]` | required | List of message dicts. Keys should match signature field names |

### Configuration

- **Frozen** (`frozen=True`) — instances are immutable after creation
- **Strict** (`extra='forbid'`) — rejects undefined fields
- **Whitespace stripped** (`str_strip_whitespace=True`)

### Message Format

Messages should use keys matching the signature fields:

```python
# Good — keys match signature fields (question, answer)
dspy.History(messages=[
    {"question": "What is DSPy?", "answer": "A framework for programming LMs."},
])

# Works but less optimal — generic role/content format
dspy.History(messages=[
    {"role": "user", "content": "What is DSPy?"},
    {"role": "assistant", "content": "A framework for programming LMs."},
])
```

## dspy.File

Added in DSPy 3.1. Wraps a file (PDF, document, etc.) as a base64 data URI following the OpenAI file content-part spec.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file_data` | `str \| None` | `None` | Data URI: `data:<mime_type>;base64,<base64_encoded_data>` |
| `file_id` | `str \| None` | `None` | Identifier for a previously uploaded file |
| `filename` | `str \| None` | `None` | Optional filename |

At least one of `file_data`, `file_id`, or `filename` must be set.

### Class Methods (preferred for creating File objects)

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_path(file_path, filename=None, mime_type=None)` | `-> File` | Read a local file, auto-detect MIME type via `mimetypes.guess_type()`, encode as base64 data URI |
| `from_bytes(file_bytes, filename=None, mime_type="application/octet-stream")` | `-> File` | Encode raw bytes as a base64 data URI |
| `from_file_id(file_id, filename=None)` | `-> File` | Reference a provider-uploaded file by ID |

## dspy.Reasoning

Added in DSPy 3.1. Captures the native reasoning/thinking trace from reasoning models as a structured, str-like output type.

### Field

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | `str` | required | The reasoning text |

The validator accepts a plain `str` (converted to `{"content": data}`), a `Reasoning` instance, or a dict with a `content` key.

### Behavior

- **String-like** — implements `__str__`, `__len__`, `__getitem__`, `__contains__`, `__iter__`, `__add__`, `__radd__`, and delegates string methods via `__getattr__`.
- `adapt_to_native_lm_feature()` sets `reasoning_effort` and removes the field from the signature when the LM supports native reasoning.
- `parse_lm_response()` extracts reasoning from a `reasoning_content` field; `parse_stream_chunk()` pulls it from streaming chunks.

When no native reasoning is available, DSPy falls back to a generated reasoning field so the same signature works on any LM.

## dspy.Tool and dspy.ToolCalls

`dspy.Tool` wraps a Python callable so the LM can invoke it as a tool; `dspy.ToolCalls` represents the LM's tool-call requests as a structured output type. Both underpin agents and ReAct. See the `/dspy-tools` skill for full API details (registration, argument schemas, execution loops).

## dspy.Example

The standard data container in DSPy. Used for training data, few-shot demos, and evaluation rows.

### Constructor

```python
dspy.Example(base=None, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base` | `dict \| Example \| None` | `None` | Dictionary or Example to copy fields from before applying kwargs |
| `**kwargs` | any | — | Field names and values (override base) |

### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `with_inputs(*keys)` | `-> Example` | Mark fields as inputs; returns new Example with `_input_keys` set. Required before use with optimizers or `dspy.Evaluate`. |
| `inputs()` | `-> Example` | Returns new Example with only the input fields |
| `labels()` | `-> Example` | Returns new Example with only the non-input (label) fields |
| `keys(include_dspy=False)` | `-> KeysView` | Returns field names (like `dict.keys()`). Pass `include_dspy=True` to include internal `dspy_` fields |
| `values(include_dspy=False)` | `-> ValuesView` | Returns field values |
| `items(include_dspy=False)` | `-> ItemsView` | Returns `(name, value)` pairs |
| `copy(**kwargs)` | `-> Example` | Shallow copy with optional field overrides |
| `without(*keys)` | `-> Example` | Returns copy with specified fields removed |
| `get(key, default=None)` | `-> Any` | Returns field value or default if missing |
| `toDict()` | `-> dict` | Converts to plain dict with recursive serialization (handles nested Examples, Pydantic models, lists) |

Fields are accessible via attribute (`ex.question`) or dictionary notation (`ex["question"]`).

## dspy.Prediction

Extends `dspy.Example`. Returned by every DSPy module forward call.

### Constructor

```python
dspy.Prediction(*args, **kwargs)
```

Inherits all `Example` parameters. Also initializes `_completions` and `_lm_usage`.

### Additional Methods (beyond Example)

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_lm_usage()` | `-> dict` | Returns token usage dict from the LM call |
| `set_lm_usage(value)` | `-> None` | Sets LM usage (used internally by the adapter) |
| `from_completions(list_or_dict, signature=None)` | `classmethod -> Prediction` | Construct from raw completion data |

Predictions with a `score` field support comparison operators (`<`, `>`, `<=`, `>=`) that compare the score as a float.

## Base Class: Type

All multimodal primitives (Image, Audio, Code, etc.) inherit from `dspy.Type` (which extends `pydantic.BaseModel`). Common inherited methods:

| Method | Description |
|--------|-------------|
| `adapt_to_native_lm_feature(signature, field_name, lm, lm_kwargs)` | Adapts signature for native LM features |
| `parse_lm_response(response)` | Parses LM response into the primitive type |
| `is_streamable()` | Returns `False` for all primitives |
