---
name: ai-summarizing
description: Condense long content into short summaries using AI. Use when summarizing meeting notes, condensing articles, creating executive briefs, extracting action items, generating TL;DRs, creating digests from long threads, summarizing customer conversations, or turning lengthy documents into bullet points. Powered by DSPy summarization.
---

# Build an AI Summarizer

Guide the user through building AI that condenses long content into useful summaries. Uses DSPy to produce consistent, faithful summaries with controllable length and detail.

## Step 1: Understand the task

Ask the user:
1. **What are you summarizing?** (meeting transcripts, articles, support threads, documents, emails?)
2. **What format should the summary be?** (bullet points, narrative paragraph, executive brief, action items?)
3. **How long should summaries be?** (one sentence, a paragraph, 3-5 bullets, custom word limit?)
4. **Who reads the summaries?** (executives, team members, customers, developers?)

## Step 2: Build a basic summarizer

### Simple text-to-summary

```python
import dspy

class Summarize(dspy.Signature):
    """Summarize the text concisely while preserving key information."""
    text: str = dspy.InputField(desc="The text to summarize")
    summary: str = dspy.OutputField(desc="A concise summary of the text")

summarizer = dspy.ChainOfThought(Summarize)
result = summarizer(text="...")
print(result.summary)
```

### Audience-aware summary

Adapt the signature for specific audiences:

```python
class SummarizeForAudience(dspy.Signature):
    """Summarize the text for the target audience."""
    text: str = dspy.InputField(desc="The text to summarize")
    audience: str = dspy.InputField(desc="Who will read this summary")
    summary: str = dspy.OutputField(desc="A summary tailored to the audience")
```

## Step 3: Structured summaries

Extract multiple aspects from the same content at once:

### Meeting transcript processor

```python
from pydantic import BaseModel, Field

class MeetingSummary(BaseModel):
    tldr: str = Field(description="One-sentence overview of the meeting")
    decisions: list[str] = Field(description="Decisions that were made")
    action_items: list[str] = Field(description="Tasks assigned with owners if mentioned")
    key_points: list[str] = Field(description="Important facts or updates discussed")

class SummarizeMeeting(dspy.Signature):
    """Extract a structured summary from a meeting transcript."""
    transcript: str = dspy.InputField(desc="Meeting transcript")
    summary: MeetingSummary = dspy.OutputField()

summarizer = dspy.ChainOfThought(SummarizeMeeting)
```

### Parallel multi-aspect extraction

Extract different aspects independently for better quality:

```python
class ExtractDecisions(dspy.Signature):
    """Extract decisions made in this meeting."""
    transcript: str = dspy.InputField()
    decisions: list[str] = dspy.OutputField(desc="Decisions that were made")

class ExtractActionItems(dspy.Signature):
    """Extract action items with assigned owners."""
    transcript: str = dspy.InputField()
    action_items: list[str] = dspy.OutputField(desc="Tasks with owners")

class ExtractKeyFacts(dspy.Signature):
    """Extract key facts and updates discussed."""
    transcript: str = dspy.InputField()
    key_facts: list[str] = dspy.OutputField(desc="Important facts and updates")

class MeetingSummarizer(dspy.Module):
    def __init__(self):
        self.tldr = dspy.ChainOfThought("transcript -> tldr")
        self.decisions = dspy.ChainOfThought(ExtractDecisions)
        self.actions = dspy.ChainOfThought(ExtractActionItems)
        self.facts = dspy.ChainOfThought(ExtractKeyFacts)

    def forward(self, transcript):
        return dspy.Prediction(
            tldr=self.tldr(transcript=transcript).tldr,
            decisions=self.decisions(transcript=transcript).decisions,
            action_items=self.actions(transcript=transcript).action_items,
            key_facts=self.facts(transcript=transcript).key_facts,
        )
```

## Step 4: Control length and detail

### Word limit enforcement

```python
class LengthControlledSummarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(SummarizeWithLimit)

    def forward(self, text, max_words=100):
        result = self.summarize(text=text, max_words=max_words)
        word_count = len(result.summary.split())
        dspy.Assert(
            word_count <= max_words,
            f"Summary is {word_count} words but must be under {max_words}. "
            "Make it more concise."
        )
        return result

class SummarizeWithLimit(dspy.Signature):
    """Summarize the text within the word limit."""
    text: str = dspy.InputField()
    max_words: int = dspy.InputField(desc="Maximum number of words for the summary")
    summary: str = dspy.OutputField(desc="A concise summary within the word limit")
```

### Detail level control

Use a detail parameter to control how much information to keep:

```python
from typing import Literal

class SummarizeWithDetail(dspy.Signature):
    """Summarize the text at the specified detail level."""
    text: str = dspy.InputField()
    detail_level: Literal["brief", "standard", "detailed"] = dspy.InputField(
        desc="brief = 1-2 sentences, standard = short paragraph, detailed = comprehensive"
    )
    summary: str = dspy.OutputField()

class MultiDetailSummarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(SummarizeWithDetail)

    def forward(self, text, detail_level="standard"):
        result = self.summarize(text=text, detail_level=detail_level)

        # Enforce approximate length expectations
        word_count = len(result.summary.split())
        limits = {"brief": 50, "standard": 150, "detailed": 400}
        max_words = limits[detail_level]

        dspy.Suggest(
            word_count <= max_words,
            f"Summary is {word_count} words for '{detail_level}' level, "
            f"aim for under {max_words}."
        )
        return result
```

## Step 5: Handle long documents

When the input is too long for a single LM call, use chunked summarization.

### Map-reduce pattern

Split → summarize each chunk → combine:

```python
class SummarizeChunk(dspy.Signature):
    """Summarize this section of a larger document."""
    chunk: str = dspy.InputField(desc="A section of a larger document")
    chunk_summary: str = dspy.OutputField(desc="Key points from this section")

class CombineSummaries(dspy.Signature):
    """Combine section summaries into one coherent summary."""
    section_summaries: list[str] = dspy.InputField(desc="Summaries of each section")
    original_length: int = dspy.InputField(desc="Word count of the original document")
    summary: str = dspy.OutputField(desc="A unified summary of the full document")

class LongDocSummarizer(dspy.Module):
    def __init__(self, chunk_size=2000):
        self.chunk_size = chunk_size
        self.map_step = dspy.ChainOfThought(SummarizeChunk)
        self.reduce_step = dspy.ChainOfThought(CombineSummaries)

    def forward(self, text):
        chunks = self._split(text)

        # Map: summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            result = self.map_step(chunk=chunk)
            chunk_summaries.append(result.chunk_summary)

        # Reduce: combine into final summary
        return self.reduce_step(
            section_summaries=chunk_summaries,
            original_length=len(text.split()),
        )

    def _split(self, text):
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunks.append(" ".join(words[i:i + self.chunk_size]))
        return chunks
```

### Hierarchical summarization

For very long documents, summarize chunks, then summarize the summaries:

```python
class HierarchicalSummarizer(dspy.Module):
    def __init__(self, chunk_size=2000, max_chunks_per_level=10):
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks_per_level
        self.summarize_chunk = dspy.ChainOfThought(SummarizeChunk)
        self.combine = dspy.ChainOfThought(CombineSummaries)

    def forward(self, text):
        chunks = self._split(text)
        summaries = [self.summarize_chunk(chunk=c).chunk_summary for c in chunks]

        # If still too many summaries, summarize again
        while len(summaries) > self.max_chunks:
            grouped = [summaries[i:i+self.max_chunks]
                       for i in range(0, len(summaries), self.max_chunks)]
            summaries = [
                self.combine(
                    section_summaries=group,
                    original_length=len(text.split()),
                ).summary
                for group in grouped
            ]

        return self.combine(
            section_summaries=summaries,
            original_length=len(text.split()),
        )

    def _split(self, text):
        words = text.split()
        return [" ".join(words[i:i+self.chunk_size])
                for i in range(0, len(words), self.chunk_size)]
```

## Step 6: Multi-format output

Generate different summary formats from the same input:

```python
class FlexibleSummarizer(dspy.Module):
    def __init__(self):
        self.bullets = dspy.ChainOfThought(BulletSummary)
        self.narrative = dspy.ChainOfThought(NarrativeSummary)
        self.executive = dspy.ChainOfThought(ExecutiveBrief)

    def forward(self, text, format="bullets"):
        if format == "bullets":
            return self.bullets(text=text)
        elif format == "narrative":
            return self.narrative(text=text)
        elif format == "executive":
            return self.executive(text=text)

class BulletSummary(dspy.Signature):
    """Summarize as a bulleted list of key points."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="Bulleted list of key points")

class NarrativeSummary(dspy.Signature):
    """Summarize as a flowing narrative paragraph."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="A narrative paragraph summary")

class ExecutiveBrief(dspy.Signature):
    """Create a brief executive summary with context, key findings, and recommendation."""
    text: str = dspy.InputField()
    context: str = dspy.OutputField(desc="One sentence of context")
    key_findings: list[str] = dspy.OutputField(desc="3-5 most important findings")
    recommendation: str = dspy.OutputField(desc="Suggested next step")
```

## Step 7: Test and optimize

### Faithfulness metric

Does the summary accurately reflect the source? No fabricated claims?

```python
class JudgeFaithfulness(dspy.Signature):
    """Judge whether the summary is faithful to the source text."""
    source_text: str = dspy.InputField()
    summary: str = dspy.InputField()
    is_faithful: bool = dspy.OutputField(desc="Does the summary only contain info from the source?")
    hallucinated_claims: list[str] = dspy.OutputField(desc="Claims not in the source, if any")

def faithfulness_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeFaithfulness)
    result = judge(source_text=example.text, summary=prediction.summary)
    return result.is_faithful
```

### Key-point coverage metric

Does the summary capture the important points?

```python
class JudgeCoverage(dspy.Signature):
    """Judge whether the summary covers the key points."""
    source_text: str = dspy.InputField()
    summary: str = dspy.InputField()
    reference_summary: str = dspy.InputField(desc="Gold-standard summary for comparison")
    coverage_score: float = dspy.OutputField(desc="0.0-1.0 how well key points are covered")

def coverage_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeCoverage)
    result = judge(
        source_text=example.text,
        summary=prediction.summary,
        reference_summary=example.summary,
    )
    return result.coverage_score
```

### Combined metric

```python
def summary_metric(example, prediction, trace=None):
    faithful = faithfulness_metric(example, prediction, trace)
    coverage = coverage_metric(example, prediction, trace)
    concise = len(prediction.summary.split()) < len(example.text.split()) * 0.3
    return (faithful * 0.4) + (coverage * 0.4) + (concise * 0.2)
```

### Optimize

```python
optimizer = dspy.BootstrapFewShot(metric=summary_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(summarizer, trainset=trainset)
```

## Key patterns

- **ChainOfThought for summaries** — reasoning helps the model decide what's important to keep
- **Pydantic models for structured summaries** — extract action items, decisions, key facts in one pass
- **Assert for length limits** — enforce word counts; DSPy retries with feedback
- **Map-reduce for long docs** — chunk, summarize each piece, combine results
- **Faithfulness metrics** — always check that summaries don't fabricate claims
- **Detail levels** — give users control over summary depth with a simple parameter

## Additional resources

- For worked examples (meetings, support threads, long docs), see [examples.md](examples.md)
- Need to extract structured fields instead of summaries? Use `/ai-parsing-data`
- Need to answer questions about docs? Use `/ai-searching-docs`
- Next: `/ai-improving-accuracy` to measure and improve your summarizer
