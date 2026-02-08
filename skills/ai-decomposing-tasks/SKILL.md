---
name: ai-decomposing-tasks
description: Break a failing complex AI task into reliable subtasks. Use when your AI works on simple inputs but fails on complex ones, extraction misses items in long documents, accuracy degrades as input grows, AI conflates multiple things at once, results are inconsistent across input types, you need to chunk long text for processing, or you want to split one unreliable AI step into multiple reliable ones.
---

# Decompose a Failing AI Task

Guide the user through splitting a single unreliable AI step into multiple reliable subtasks. The insight: when a single prompt fails on complex inputs, restructuring the task — not just tweaking the prompt — is often the fix.

## Step 1: Diagnose why single-step fails

Ask the user:
1. **What's the task?** (extraction, classification, generation, etc.)
2. **When does it work?** (simple inputs, short text, single items)
3. **When does it fail?** (long documents, many items, mixed formats)

### Common failure modes

Look at the errors. They usually fall into one of these patterns:

| Failure mode | What you see | Root cause |
|-------------|-------------|------------|
| Missed items | Extracts 3 of 7 line items | Input overwhelms the context — too much to track at once |
| Conflated fields | Mixes up sender/recipient addresses | Multiple similar things extracted simultaneously |
| Inconsistent results | Works on invoice A, fails on invoice B | Different input formats need different handling |
| Degraded accuracy | 95% on short text, 60% on long text | Input length exceeds what a single pass can reliably process |

If the task works on simple inputs but fails on complex ones, decomposition is the right lever. If it fails on *everything*, try `/ai-improving-accuracy` first.

## Step 2: Choose a decomposition strategy

Match the failure mode to a pattern:

```
What's going wrong?
|
+- Input is too long, AI loses focus
|  → Chunk-then-process (Step 3)
|
+- AI conflates multiple similar things
|  → Sequential extraction (Step 4)
|
+- AI misses items in variable-length lists
|  → Identify-then-process (Step 5)
|
+- Different input types need different handling
|  → Classify-then-specialize (see /ai-building-pipelines)
```

You can combine strategies. A long document with variable-length lists might need chunking *and* identify-then-process.

## Step 3: Chunk-then-process

Split long input into overlapping chunks, process each, then deduplicate results.

**When to use:** Input exceeds what the model can reliably process in one pass. Typical signs: accuracy drops sharply as input length grows.

```python
import dspy
from pydantic import BaseModel, Field

class ExtractedItem(BaseModel):
    name: str
    value: str
    source_text: str = Field(description="The exact text this was extracted from")

class ExtractFromChunk(dspy.Signature):
    """Extract all relevant items from this section of the document."""
    chunk: str = dspy.InputField(desc="A section of the document")
    items: list[ExtractedItem] = dspy.OutputField(desc="All items found in this section")

class ChunkAndExtract(dspy.Module):
    def __init__(self, chunk_size=2000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.extract = dspy.ChainOfThought(ExtractFromChunk)

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks at paragraph boundaries."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - self.overlap
        return chunks

    def _deduplicate(self, all_items: list[ExtractedItem]) -> list[ExtractedItem]:
        """Remove duplicate extractions from overlapping chunks."""
        seen = set()
        unique = []
        for item in all_items:
            key = (item.name.lower().strip(), item.value.lower().strip())
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def forward(self, document: str):
        chunks = self._chunk_text(document)
        all_items = []

        for chunk in chunks:
            result = self.extract(chunk=chunk)
            all_items.extend(result.items)

        unique_items = self._deduplicate(all_items)
        return dspy.Prediction(items=unique_items)
```

Key details:
- **Overlap** prevents items at chunk boundaries from being split and missed
- **Paragraph-aware splitting** is better than raw character splitting — try to break at `\n\n` boundaries
- **Deduplication** is essential because overlapping chunks will extract the same items twice
- Include `source_text` in the output so you can trace extractions back to the document

## Step 4: Sequential extraction (the Salomatic pattern)

Extract one thing first, then use that result to constrain the next extraction. This is the pattern that took a medical report system from 40% error rate to near-zero.

**When to use:** The AI conflates multiple similar things, or extracting everything at once overwhelms it.

```python
class IdentifyPanels(dspy.Signature):
    """Identify all lab test panels in the medical report."""
    report: str = dspy.InputField(desc="Medical lab report")
    panel_names: list[str] = dspy.OutputField(desc="Names of all test panels found")

class LabResult(BaseModel):
    test_name: str
    value: str
    unit: str
    reference_range: str
    flag: str = Field(description="'normal', 'high', or 'low'")

class ExtractPanelResults(dspy.Signature):
    """Extract all test results for a specific panel from the report."""
    report: str = dspy.InputField(desc="Medical lab report")
    panel_name: str = dspy.InputField(desc="The specific panel to extract results for")
    results: list[LabResult] = dspy.OutputField(desc="All test results for this panel")

class SequentialExtractor(dspy.Module):
    def __init__(self):
        self.identify = dspy.ChainOfThought(IdentifyPanels)
        self.extract = dspy.ChainOfThought(ExtractPanelResults)

    def forward(self, report: str):
        # Step 1: Identify what's in the report
        panels = self.identify(report=report)

        dspy.Assert(
            len(panels.panel_names) > 0,
            "No test panels found — is this a valid lab report?"
        )

        # Step 2: Extract results per panel
        all_results = {}
        for panel_name in panels.panel_names:
            result = self.extract(report=report, panel_name=panel_name)
            all_results[panel_name] = result.results

        return dspy.Prediction(
            panels=panels.panel_names,
            results=all_results,
        )
```

Why this works:
- **Step 1 is easy** — just identify panel names (low cognitive load)
- **Step 2 is focused** — extract results for *one specific panel* at a time
- The model doesn't have to juggle "find all panels AND extract all results" simultaneously
- Each extraction is scoped to a smaller, well-defined subtask

This same pattern applies beyond medical reports — any time you're extracting multiple groups of similar things (invoice sections, resume sections, contract clauses).

## Step 5: Identify-then-process

First count or name the items, then process each one individually. This prevents the "missed items" failure where the model extracts 3 of 7 items.

**When to use:** Variable-length lists where the model consistently misses items.

```python
class IdentifyLineItems(dspy.Signature):
    """Identify all line items in the invoice. List every item, even small ones."""
    invoice_text: str = dspy.InputField(desc="Raw invoice text")
    item_descriptions: list[str] = dspy.OutputField(
        desc="Brief description of each line item, in order they appear"
    )

class LineItemDetail(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class ExtractLineItem(dspy.Signature):
    """Extract the details for a specific line item from the invoice."""
    invoice_text: str = dspy.InputField(desc="Raw invoice text")
    item_description: str = dspy.InputField(desc="The specific item to extract details for")
    details: LineItemDetail = dspy.OutputField()

class IdentifyThenExtract(dspy.Module):
    def __init__(self):
        self.identify = dspy.ChainOfThought(IdentifyLineItems)
        self.extract_item = dspy.ChainOfThought(ExtractLineItem)

    def forward(self, invoice_text: str):
        # Step 1: Identify all items (just names — low cognitive load)
        items = self.identify(invoice_text=invoice_text)

        dspy.Assert(
            len(items.item_descriptions) > 0,
            "No line items found in invoice"
        )

        # Step 2: Extract details per item
        line_items = []
        for desc in items.item_descriptions:
            result = self.extract_item(
                invoice_text=invoice_text,
                item_description=desc,
            )
            line_items.append(result.details)

        return dspy.Prediction(line_items=line_items)
```

The identify step works as an "attention anchor" — once the model has listed all items, the extraction step knows exactly what to look for and is much less likely to skip anything.

## Step 6: Compare single-step vs decomposed

Always measure the improvement. The decomposed version costs more (multiple LM calls), so you need to verify the accuracy gain justifies the cost:

```python
from dspy.evaluate import Evaluate

# Build both versions
single_step = dspy.ChainOfThought(ExtractAllItems)  # Original single-step
decomposed = IdentifyThenExtract()  # Decomposed version

def extraction_metric(example, prediction, trace=None):
    """Measure recall — what fraction of gold items were extracted."""
    gold_items = set(item.lower() for item in example.item_names)
    pred_items = set(item.description.lower() for item in prediction.line_items)
    if not gold_items:
        return 1.0
    return len(gold_items & pred_items) / len(gold_items)

evaluator = Evaluate(devset=devset, metric=extraction_metric, num_threads=4, display_table=5)

# Compare
single_score = evaluator(single_step)
decomposed_score = evaluator(decomposed)

print(f"Single-step: {single_score:.1f}%")
print(f"Decomposed:  {decomposed_score:.1f}%")
```

### Stratify by complexity

The real value of decomposition shows on complex inputs. Measure separately:

```python
simple_devset = [ex for ex in devset if len(ex.item_names) <= 3]
complex_devset = [ex for ex in devset if len(ex.item_names) > 3]

simple_evaluator = Evaluate(devset=simple_devset, metric=extraction_metric)
complex_evaluator = Evaluate(devset=complex_devset, metric=extraction_metric)

print("Simple inputs:")
print(f"  Single-step: {simple_evaluator(single_step):.1f}%")
print(f"  Decomposed:  {simple_evaluator(decomposed):.1f}%")

print("Complex inputs:")
print(f"  Single-step: {complex_evaluator(single_step):.1f}%")
print(f"  Decomposed:  {complex_evaluator(decomposed):.1f}%")
```

If the decomposed version doesn't significantly outperform on complex inputs, you may not need the decomposition. Stick with the simpler single-step approach.

## Step 7: Optimize end-to-end

MIPROv2 can optimize all stages of your decomposed pipeline together. This is powerful because the identify step learns to produce outputs that help the extract step:

```python
optimizer = dspy.MIPROv2(metric=extraction_metric, auto="medium")
optimized = optimizer.compile(decomposed, trainset=trainset)

# Verify improvement
optimized_score = evaluator(optimized)
print(f"Decomposed (unoptimized): {decomposed_score:.1f}%")
print(f"Decomposed (optimized):   {optimized_score:.1f}%")
```

### Use different models per stage

The identify step (listing items) is simpler than the extract step (pulling details). Use a cheaper model for the easy step:

```python
cheap_lm = dspy.LM("openai/gpt-4o-mini")
quality_lm = dspy.LM("openai/gpt-4o")

decomposed.identify.set_lm(cheap_lm)      # Cheap for listing
decomposed.extract_item.set_lm(quality_lm) # Quality for extraction
```

See `/ai-cutting-costs` for more cost strategies.

## Key patterns

- **Decompose when complexity causes failures** — if it works on simple inputs but fails on complex ones, restructure
- **Identify-then-process prevents missed items** — listing first creates an attention anchor
- **Sequential extraction prevents conflation** — extract one type of thing at a time
- **Chunking handles long documents** — overlap chunks and deduplicate results
- **Always compare against single-step** — decomposition costs more, so verify the accuracy gain
- **Stratify by complexity** — the payoff shows on complex inputs, not simple ones
- **Optimize end-to-end** — MIPROv2 tunes all stages together for best results
- **Cheap models for easy stages** — the identify step rarely needs an expensive model

## Additional resources

- For worked examples (medical reports, invoices, resumes), see [examples.md](examples.md)
- Already know your pipeline stages? Use `/ai-building-pipelines` to wire them together
- Need to improve accuracy within a single step? Use `/ai-improving-accuracy`
- Need to extract structured data? Start with `/ai-parsing-data` — decompose only if it struggles on complex inputs
- Next: `/ai-improving-accuracy` to measure and optimize your decomposed pipeline
