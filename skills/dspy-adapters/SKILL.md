---
name: dspy-adapters
description: Use when you need to customize how DSPy formats prompts for a specific provider — switching from chat to completion format, forcing JSON output, or debugging prompt rendering issues. Common scenarios: debugging why your prompt looks wrong when sent to the model, switching from OpenAI to Anthropic and the formatting breaks, forcing the model to return valid JSON instead of markdown, working with completion-style models that don't support chat format, customizing system messages, or handling models that choke on structured output instructions. Related: ai-switching-models, ai-following-rules, ai-parsing-data. Also: prompt template rendering, how DSPy builds the prompt, custom system message in DSPy, JSON mode not working, model ignores format instructions, switch from chat to completion API, dspy.ChatAdapter, dspy.JSONAdapter, prompt formatting issues, debug what DSPy sends to the model.
---

# Control Prompt Formatting with DSPy Adapters

Adapters sit between your DSPy modules and the language model. They control how signatures get turned into prompts and how LM responses get parsed back into typed Python objects. Most of the time the default adapter just works -- but when you need tighter control over structured output, or you are working with reasoning models that struggle with formatting, adapters give you that control.

## What adapters do

Every time a DSPy module calls an LM, an adapter handles two jobs:

1. **Format** -- Convert the signature, few-shot demos, and inputs into a prompt (system message + user/assistant messages).
2. **Parse** -- Extract the output fields from the LM's raw text response and cast them to the declared Python types.

You never call adapters directly. You configure one globally or per-module, and DSPy uses it behind the scenes.

## The three built-in adapters

| Adapter | How it formats | How it parses | Best for |
|---------|---------------|---------------|----------|
| `ChatAdapter` | Field markers like `[[ ## field_name ## ]]` | Splits on field headers | General use (default) |
| `JSONAdapter` | Requests JSON output with field schema | `json_repair` + type casting | Reliable structured output |
| `TwoStepAdapter` | Natural language prompt (no formatting constraints) | Sends raw response to a second LM for extraction | Reasoning models (o1, o3) |

## ChatAdapter (the default)

`ChatAdapter` is what DSPy uses unless you say otherwise. It formats prompts with field delimiters and parses responses by looking for those same delimiters in the output.

```python
import dspy

# ChatAdapter is used automatically -- no configuration needed
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

classify = dspy.ChainOfThought("text -> label")
result = classify(text="Great product!")
print(result.label)
```

### Constructor parameters

```python
dspy.ChatAdapter(
    callbacks=None,                    # Optional list of BaseCallback objects
    use_native_function_calling=False, # Use native function calling features
    native_response_types=None,        # Output field types handled natively by the LM
    use_json_adapter_fallback=True,    # Fall back to JSONAdapter on parse failure
)
```

### Key behavior

- Formats each field with `[[ ## field_name ## ]]` delimiters in the prompt.
- Few-shot demos become alternating user/assistant message pairs.
- If parsing fails, it automatically retries using `JSONAdapter` (unless you set `use_json_adapter_fallback=False`).
- Works well with most models out of the box.

### When to use ChatAdapter explicitly

You rarely need to instantiate `ChatAdapter` yourself. Do it when you want to:

- Disable the JSON fallback: `dspy.ChatAdapter(use_json_adapter_fallback=False)`
- Enable native function calling: `dspy.ChatAdapter(use_native_function_calling=True)`

```python
adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
dspy.configure(lm=lm, adapter=adapter)
```

## JSONAdapter

`JSONAdapter` extends `ChatAdapter` and instructs the LM to respond with a JSON object matching your output fields. It uses the provider's native structured output mode when available, falling back to `response_format: {"type": "json_object"}`.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
adapter = dspy.JSONAdapter()
dspy.configure(lm=lm, adapter=adapter)

# Now all modules request JSON output from the LM
classify = dspy.Predict("text -> label: str, confidence: float")
result = classify(text="Great product!")
print(result.label)       # positive
print(result.confidence)  # 0.95
```

### Constructor parameters

```python
dspy.JSONAdapter(
    callbacks=None,                    # Optional list of BaseCallback objects
    use_native_function_calling=True,  # Enabled by default (unlike ChatAdapter)
)
```

### Key behavior

- Tells the LM to respond with a JSON object whose keys match the output field names.
- Includes type hints in the prompt (e.g., `"(must be formatted as a valid Python str)"`).
- Parses responses with `json_repair` for resilience against minor formatting errors.
- Validates that all required output fields are present and casts values to their annotated types.
- Raises `AdapterParseError` if the response cannot be parsed as a JSON object.

### When to use JSONAdapter

Use `JSONAdapter` when you need more reliable structured output, especially with:

- Complex Pydantic output types (nested models, lists of objects)
- Models that sometimes break the `[[ ## field ## ]]` format
- Applications where parse failures are costly (production APIs, batch pipelines)

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    line_items: list[dict]

class ExtractInvoice(dspy.Signature):
    """Extract invoice details from the document text."""
    document: str = dspy.InputField()
    invoice: Invoice = dspy.OutputField()

lm = dspy.LM("openai/gpt-4o")
adapter = dspy.JSONAdapter()
dspy.configure(lm=lm, adapter=adapter)

extractor = dspy.Predict(ExtractInvoice)
result = extractor(document="Invoice #1234 from Acme Corp. Total: $1,250.00 ...")
print(result.invoice.vendor)      # Acme Corp
print(result.invoice.total)       # 1250.0
print(result.invoice.line_items)  # [...]
```

## TwoStepAdapter

`TwoStepAdapter` is designed for reasoning models (like OpenAI's o1 and o3 series) that produce better answers when they are not forced into a rigid output format. It splits the work into two steps:

1. **Step 1** -- The main LM gets a natural language prompt with no formatting constraints. It can think freely.
2. **Step 2** -- A smaller, cheaper extraction model reads the main LM's response and extracts the structured output fields.

```python
import dspy

# Main LM: a reasoning model that struggles with structured output
main_lm = dspy.LM("openai/o3-mini", max_tokens=16000, temperature=1.0)

# Extraction model: a fast, cheap model for parsing
extraction_lm = dspy.LM("openai/gpt-4o-mini")

adapter = dspy.TwoStepAdapter(extraction_model=extraction_lm)
dspy.configure(lm=main_lm, adapter=adapter)

# The reasoning model thinks freely; gpt-4o-mini extracts the answer
solver = dspy.ChainOfThought("question -> answer")
result = solver(question="What is the sum of the first 100 prime numbers?")
print(result.answer)
```

### Constructor parameters

```python
dspy.TwoStepAdapter(
    extraction_model=dspy.LM("openai/gpt-4o-mini"),  # Required: LM for structured extraction
)
```

### Key behavior

- The main LM receives a simplified, natural language prompt -- no field delimiters or JSON instructions.
- The extraction model uses `ChatAdapter` internally to parse the main LM's freeform response into structured fields.
- Adds cost (two LM calls per prediction) but improves quality for reasoning models.

### When to use TwoStepAdapter

Use `TwoStepAdapter` when:

- Your main LM is a reasoning model (o1, o3, o3-mini) that performs worse when forced to follow formatting rules.
- You want the best reasoning quality and can tolerate the extra latency/cost of a second LM call.
- The extraction step is straightforward (the reasoning is the hard part, not the formatting).

## Decision table: which adapter to use

| Situation | Adapter | Why |
|-----------|---------|-----|
| General use, most models | `ChatAdapter` (default) | Works out of the box, no config needed |
| Need reliable JSON/Pydantic output | `JSONAdapter` | Stricter parsing, native structured output support |
| Complex nested output types | `JSONAdapter` | Better at complex schemas than field-delimiter parsing |
| Using reasoning models (o1, o3) | `TwoStepAdapter` | Reasoning models perform worse with format constraints |
| Parse failures in production | `JSONAdapter` | More resilient parsing with `json_repair` |
| Fastest iteration, prototyping | `ChatAdapter` (default) | Zero config, good enough for most tasks |

## Configuring adapters

### Global configuration

Set the adapter for all modules at once:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
adapter = dspy.JSONAdapter()
dspy.configure(lm=lm, adapter=adapter)
```

### Temporary override with dspy.context

Switch adapters for a block of code without changing the global config:

```python
json_adapter = dspy.JSONAdapter()

# Use JSONAdapter just for this block
with dspy.context(adapter=json_adapter):
    result = extractor(document=text)

# Back to the default ChatAdapter outside the block
```

### Per-module adapter assignment

Assign different adapters to different modules using `set_adapter()`:

```python
class Pipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("text -> label")
        self.extract = dspy.Predict(ExtractDetails)

    def forward(self, text):
        # Classification is simple -- default ChatAdapter is fine
        label = self.classify(text=text)
        return self.extract(text=text)

pipeline = Pipeline()

# Use JSONAdapter only for the extraction step
pipeline.extract.set_adapter(dspy.JSONAdapter())
```

## Custom adapters

You can build your own adapter by subclassing the base `Adapter` class. This is advanced -- only needed if the built-in adapters do not fit your use case.

```python
import dspy
from dspy.adapters import Adapter

class MyAdapter(Adapter):
    def format(self, signature, demos, inputs, messages=None):
        """Convert signature + inputs into a list of messages for the LM."""
        system_msg = {"role": "system", "content": f"Task: {signature.instructions}"}
        user_msg = {"role": "user", "content": str(inputs)}
        return [system_msg, user_msg]

    def parse(self, signature, completion):
        """Extract output fields from the LM's raw response text."""
        # Your custom parsing logic here
        fields = {}
        for field_name in signature.output_fields:
            fields[field_name] = completion.strip()
        return fields

adapter = MyAdapter()
dspy.configure(lm=lm, adapter=adapter)
```

Override `format()` to control how prompts are built and `parse()` to control how responses are read. The return from `parse()` should be a dict mapping output field names to their values.

## Common patterns

### Fallback chain: ChatAdapter with JSON fallback (default)

By default, `ChatAdapter` already falls back to `JSONAdapter` on parse failure. This gives you the best of both worlds -- fast field-delimiter parsing most of the time, with JSON as a safety net.

```python
# This is the default behavior -- you get it for free
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
# ChatAdapter is used, with automatic JSONAdapter fallback on failure
```

To disable this fallback:

```python
adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
dspy.configure(lm=lm, adapter=adapter)
```

### Reasoning model setup

```python
# Full setup for reasoning models
reasoning_lm = dspy.LM("openai/o3-mini", max_tokens=16000, temperature=1.0)
extraction_lm = dspy.LM("openai/gpt-4o-mini")

adapter = dspy.TwoStepAdapter(extraction_model=extraction_lm)
dspy.configure(lm=reasoning_lm, adapter=adapter)
```

### Mixed adapter pipeline

```python
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)  # default ChatAdapter

class MixedPipeline(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought("document -> summary")
        self.extract = dspy.Predict(ExtractInvoice)

    def forward(self, document):
        summary = self.summarize(document=document)
        return self.extract(document=document)

pipeline = MixedPipeline()

# ChatAdapter for summarization (freeform text is fine)
# JSONAdapter for extraction (need reliable structured output)
pipeline.extract.set_adapter(dspy.JSONAdapter())
```

## Cross-references

- **`/dspy-signatures`** -- Define the input/output fields that adapters format and parse
- **`/dspy-lm`** -- Configure the language model that adapters communicate with
- **`/dspy-modules`** -- Modules that use adapters under the hood (Predict, ChainOfThought, etc.)
- Not sure which skill to use next? Try `/ai-do` to get routed to the right one
