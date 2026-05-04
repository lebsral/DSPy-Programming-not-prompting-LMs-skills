# Citations API Reference

> Condensed from [dspy.ai/api/experimental/Citations](https://dspy.ai/api/experimental/Citations/). Verify against upstream for latest.

## Citations type

```python
from dspy.experimental import Citations
```

A list-like container of citation objects. Used as an output field type in DSPy signatures.

**Usage in signatures:**

```python
class MySignature(dspy.Signature):
    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: Citations = dspy.OutputField()
```

**Iteration:**

```python
for citation in result.citations:
    print(citation.cited_text, citation.document_index)
```

### Citation object fields

| Field | Type | Description |
|-------|------|-------------|
| `cited_text` | `str` | The exact text passage being cited |
| `document_index` | `int` | Index into the context list identifying the source |
| `start` | `int` | Start character offset in the source document (native mode) |
| `end` | `int` | End character offset in the source document (native mode) |

## Document type

```python
from dspy.experimental import Document

doc = Document(
    text="...",         # str -- required, document content
    title="...",        # str -- optional, document title
    source_id="...",    # str -- optional, unique identifier
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `str` | required | Full text content of the document |
| `title` | `str` | `""` | Human-readable document title |
| `source_id` | `str` | `""` | Unique identifier for the document |

## Native Anthropic Citations

Enable via adapter configuration:

```python
dspy.configure(
    lm=dspy.LM("anthropic/claude-sonnet-4-5-20250929"),
    adapter=dspy.ChatAdapter(
        adapt_to_native_lm_feature=["citations"],
    ),
)
```

When enabled:
- Anthropic's Citations API extracts citations during generation
- `cited_text` exactly matches source text (character-level)
- `start` and `end` offsets are populated
- Higher accuracy than prompt-based extraction

**Only works with Anthropic models.** Other providers fall back to prompt-based citation extraction.

## Class methods

| Method | Description |
|--------|-------------|
| `Citations.from_dict_list(dicts)` | Create from list of citation dictionaries |
| `Citations.parse_lm_response(response)` | Extract citations from raw LM response |
| `Citations.format(citations)` | Format citations for display |

## Provider compatibility

| Provider | Native citations | Prompt-based fallback |
|----------|-----------------|----------------------|
| Anthropic (Claude) | Yes (via adapt_to_native_lm_feature) | Yes |
| OpenAI | No | Yes |
| Local models | No | Yes |
