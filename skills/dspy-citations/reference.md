# Citations API Reference

> Condensed from [dspy.ai/api/experimental/Citations](https://dspy.ai/api/experimental/Citations/). Verify against upstream for latest.

## Citations type

```python
from dspy.experimental import Citations
```

A container holding citation objects under its `.citations` attribute. Used as an output
field type in DSPy signatures.

**Usage in signatures:**

```python
from dspy.experimental import Citations, Document

class MySignature(dspy.Signature):
    documents: list[Document] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: Citations = dspy.OutputField()
```

**Iteration** (note the nested `.citations`):

```python
for citation in result.citations.citations:
    print(citation.cited_text, citation.document_index)
```

### Citation object fields

| Field | Type | Description |
|-------|------|-------------|
| `cited_text` | `str` | The exact text passage being cited |
| `document_index` | `int` | Index into the input document list identifying the source |
| `document_title` | `str` | Title of the cited document |
| `start_char_index` | `int` | Start character offset in the source document |
| `end_char_index` | `int` | End character offset in the source document |
| `supported_text` | `str` | The text in the answer that this citation supports |

## Document type

```python
from dspy.experimental import Document

doc = Document(
    data="...",                  # str -- required, document content
    title="...",                 # str -- optional, document title
    media_type="text/plain",     # str -- optional, content media type
    context="...",               # str -- optional, extra context for the model
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data` | `str` | required | Full text content of the document |
| `title` | `str` | `None` | Human-readable document title |
| `media_type` | `str` | `"text/plain"` | Media type of the document content |
| `context` | `str` | `None` | Optional extra context passed to the model |

## Native Anthropic Citations

Native citations are **automatic** -- there is no adapter kwarg to set. Just configure an
Anthropic LM, declare a `Citations` output field, and pass `documents: list[Document]`:

```python
dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-4-5-20250929"))
# A signature with a Citations output field + documents input now uses
# Anthropic's native Citations API automatically.
```

When using an Anthropic model:
- Anthropic's Citations API extracts citations during generation
- `cited_text` exactly matches source text (character-level)
- `start_char_index` and `end_char_index` offsets are populated
- Higher accuracy than prompt-based extraction

**Only works with Anthropic models.** Other providers fall back to prompt-based citation extraction.

## Methods

| Method | Description |
|--------|-------------|
| `citation.format()` | Format a single citation for display (user-facing) |

`Citations.parse_lm_response(...)` and `ChatAdapter.adapt_to_native_lm_feature(...)` are
internal hooks DSPy uses to wire up the native Anthropic Citations feature. You do not call
them directly -- declaring a `Citations` output field with an Anthropic LM is enough.

## Provider compatibility

| Provider | Native citations | Prompt-based fallback |
|----------|-----------------|----------------------|
| Anthropic (Claude) | Yes (automatic with a `Citations` output field) | Yes |
| OpenAI | No | Yes |
| Local models | No | Yes |
