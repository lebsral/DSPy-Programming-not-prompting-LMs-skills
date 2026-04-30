# dspy-chatadapter Examples

## Example 1: Debugging a parse failure

A common scenario: your DSPy module works with one model but breaks with another because the new model does not follow the field delimiter format consistently.

```python
import dspy

# Works fine with GPT-4o-mini
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

class ExtractContact(dspy.Signature):
    """Extract contact information from the text."""
    text: str = dspy.InputField()
    name: str = dspy.OutputField()
    email: str = dspy.OutputField()
    phone: str = dspy.OutputField()

extractor = dspy.Predict(ExtractContact)
result = extractor(text="Call John Smith at john@acme.com or 555-0123")

# Step 1: Inspect what ChatAdapter sent and received
dspy.inspect_history(n=1)
# Look for the [[ ## field ## ]] delimiters in the prompt and response.
# If the response lacks delimiters, the model is ignoring the format.

# Step 2: Check if JSON fallback kicked in
# Run the same call again and inspect history for TWO consecutive calls.
# If you see a second call with JSON instructions, fallback triggered automatically.

# Step 3: If fallback also fails, switch to JSONAdapter explicitly
adapter = dspy.JSONAdapter()
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=adapter)

result = extractor(text="Call John Smith at john@acme.com or 555-0123")
print(f"{result.name}, {result.email}, {result.phone}")
```

What this demonstrates:

- **`dspy.inspect_history()`** is the primary tool for debugging adapter behavior
- **JSON fallback** often self-heals parse failures without any code changes
- **Switching to JSONAdapter** is the fix when ChatAdapter's delimiter format consistently fails with a specific model

## Example 2: Mixed adapter pipeline with per-module assignment

A pipeline where summarization uses ChatAdapter (freeform text is fine) but data extraction uses JSONAdapter (need reliable structured output).

```python
import dspy
from pydantic import BaseModel

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

# --- Pydantic model for structured extraction ---
class MeetingAction(BaseModel):
    assignee: str
    task: str
    deadline: str

class MeetingExtraction(dspy.Signature):
    """Extract action items from meeting notes."""
    notes: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="2-3 sentence summary")
    action_items: list[MeetingAction] = dspy.OutputField()

# --- Pipeline ---
class MeetingProcessor(dspy.Module):
    def __init__(self):
        # Simple summarization -- ChatAdapter is fine
        self.summarize = dspy.ChainOfThought("notes -> summary")
        # Structured extraction -- JSONAdapter for reliability
        self.extract = dspy.Predict(MeetingExtraction)

    def forward(self, notes):
        summary = self.summarize(notes=notes)
        extraction = self.extract(notes=notes)
        return extraction

processor = MeetingProcessor()

# Assign JSONAdapter only to the extraction step
processor.extract.set_adapter(dspy.JSONAdapter())

result = processor(notes="""
Q3 planning meeting, Oct 15.
- Sarah will finalize the budget by Oct 20.
- Mike to hire 2 engineers by end of November.
- Team agreed to ship v2.0 by December 1.
""")

print(f"Summary: {result.summary}")
for item in result.action_items:
    print(f"  {item.assignee}: {item.task} (by {item.deadline})")
```

What this demonstrates:

- **`set_adapter()`** assigns a different adapter to a specific module without changing the global config
- **ChatAdapter for freeform output** (summaries, reasoning) where exact formatting does not matter
- **JSONAdapter for structured output** (Pydantic models, lists of objects) where parse reliability matters
- **No global adapter change needed** -- only the module that needs stricter parsing gets JSONAdapter

## Example 3: Generating fine-tuning data from a DSPy program

Export the exact prompt format ChatAdapter uses so you can fine-tune a model that responds correctly to it.

```python
import json
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

# --- Define signature and collect examples ---
class ClassifyTicket(dspy.Signature):
    """Classify a support ticket by category and urgency."""
    ticket_text: str = dspy.InputField()
    category: str = dspy.OutputField()
    urgency: str = dspy.OutputField()

# Labeled examples for fine-tuning
examples = [
    {"inputs": {"ticket_text": "Payment failed twice, charged both times"},
     "outputs": {"category": "billing", "urgency": "high"}},
    {"inputs": {"ticket_text": "How do I change my email address?"},
     "outputs": {"category": "account", "urgency": "low"}},
    {"inputs": {"ticket_text": "App crashes when I upload photos"},
     "outputs": {"category": "bug", "urgency": "medium"}},
    # ... hundreds more examples
]

# --- Generate OpenAI fine-tuning JSONL ---
adapter = dspy.ChatAdapter()

with open("finetune_data.jsonl", "w") as f:
    for ex in examples:
        finetune_record = adapter.format_finetune_data(
            signature=ClassifyTicket,
            demos=[],  # no few-shot demos in fine-tuning data
            inputs=ex["inputs"],
            outputs=ex["outputs"],
        )
        # Each record is {"messages": [system, user, assistant]}
        f.write(json.dumps(finetune_record) + "\n")

print(f"Wrote {len(examples)} fine-tuning examples to finetune_data.jsonl")

# The fine-tuned model will respond using [[ ## field ## ]] delimiters
# that ChatAdapter can parse natively -- no adapter changes needed.
```

What this demonstrates:

- **`format_finetune_data()`** produces OpenAI-compatible `{"messages": [...]}` format
- **Delimiter consistency** -- the fine-tuned model learns to respond with `[[ ## field ## ]]` headers that ChatAdapter already knows how to parse
- **No adapter switch needed** after fine-tuning -- the model speaks ChatAdapter's format natively
- **Production pattern** for teams using `dspy.BootstrapFinetune` who want to understand what data format is generated
