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

## Base Class: Type

All primitives inherit from `dspy.Type` (which extends `pydantic.BaseModel`). Common inherited methods:

| Method | Description |
|--------|-------------|
| `adapt_to_native_lm_feature(signature, field_name, lm, lm_kwargs)` | Adapts signature for native LM features |
| `parse_lm_response(response)` | Parses LM response into the primitive type |
| `is_streamable()` | Returns `False` for all primitives |
