# AI Summarizing — Worked Examples

## Example 1: Meeting transcript processor

Extract action items, decisions, and follow-ups from meeting transcripts.

### Setup

```python
import dspy
from pydantic import BaseModel, Field

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
```

### Signatures and module

```python
class MeetingOutput(BaseModel):
    tldr: str = Field(description="One-sentence meeting summary")
    decisions: list[str] = Field(description="Decisions that were made")
    action_items: list[str] = Field(
        description="Tasks with owner and deadline if mentioned, e.g. 'Alice: update pricing page by Friday'"
    )
    follow_ups: list[str] = Field(description="Topics to revisit or discuss later")

class ProcessMeeting(dspy.Signature):
    """Extract a structured summary from a meeting transcript."""
    transcript: str = dspy.InputField(desc="Meeting transcript text")
    output: MeetingOutput = dspy.OutputField()

class MeetingProcessor(dspy.Module):
    def __init__(self):
        self.process = dspy.ChainOfThought(ProcessMeeting)

    def forward(self, transcript):
        result = self.process(transcript=transcript)

        # Ensure we captured at least something
        dspy.Suggest(
            len(result.output.action_items) > 0,
            "Most meetings have at least one action item — look again"
        )
        dspy.Suggest(
            len(result.output.decisions) > 0,
            "Check if any decisions were made, even implicit ones"
        )

        return result
```

### Usage

```python
processor = MeetingProcessor()

result = processor(transcript="""
Alice: Let's discuss the Q3 roadmap. We need to decide on the pricing change.
Bob: I think we should go with the 15% increase for enterprise tier.
Alice: Agreed. Bob, can you update the pricing page by Friday?
Carol: I'll prepare the customer communication. Should be ready by next Wednesday.
Alice: Great. We should also revisit the free tier limits next month.
Bob: One more thing — the API latency issue. Carol, can you look into that?
Carol: Sure, I'll investigate by end of week.
""")

print(result.output.tldr)
# "Team agreed on 15% enterprise pricing increase and assigned tasks for pricing page, customer comms, and API latency."
print(result.output.decisions)
# ["15% price increase for enterprise tier"]
print(result.output.action_items)
# ["Bob: update pricing page by Friday", "Carol: prepare customer communication by Wednesday", "Carol: investigate API latency by end of week"]
print(result.output.follow_ups)
# ["Revisit free tier limits next month"]
```

### Metric and optimization

```python
def meeting_metric(example, prediction, trace=None):
    """Score based on action item and decision coverage."""
    score = 0.0

    # Check action items coverage
    pred_actions = set(a.lower() for a in prediction.output.action_items)
    gold_actions = set(a.lower() for a in example.output.action_items)
    if gold_actions:
        action_overlap = len(pred_actions & gold_actions) / len(gold_actions)
        score += 0.5 * action_overlap

    # Check decisions coverage
    pred_decisions = set(d.lower() for d in prediction.output.decisions)
    gold_decisions = set(d.lower() for d in example.output.decisions)
    if gold_decisions:
        decision_overlap = len(pred_decisions & gold_decisions) / len(gold_decisions)
        score += 0.3 * decision_overlap

    # TLDR exists and is short
    if prediction.output.tldr and len(prediction.output.tldr.split()) < 30:
        score += 0.2

    return score

optimizer = dspy.BootstrapFewShot(metric=meeting_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(MeetingProcessor(), trainset=trainset)
```

---

## Example 2: Customer support thread summarizer

Condense long support conversations into a status summary for handoffs.

### Signatures and module

```python
class ThreadSummary(BaseModel):
    issue: str = Field(description="What the customer's problem is")
    status: str = Field(description="Current status: resolved, pending, escalated")
    steps_taken: list[str] = Field(description="What's been tried so far")
    next_step: str = Field(description="What needs to happen next")

class SummarizeThread(dspy.Signature):
    """Summarize a customer support thread for agent handoff."""
    thread: str = dspy.InputField(desc="The full support conversation")
    summary: ThreadSummary = dspy.OutputField()

class SupportSummarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(SummarizeThread)

    def forward(self, thread):
        result = self.summarize(thread=thread)

        dspy.Assert(
            result.summary.status in ["resolved", "pending", "escalated"],
            "Status must be one of: resolved, pending, escalated"
        )
        dspy.Suggest(
            len(result.summary.steps_taken) > 0,
            "There should be at least one step taken in the conversation"
        )

        return result
```

### Usage

```python
summarizer = SupportSummarizer()

result = summarizer(thread="""
Customer: My invoice #4521 shows the wrong amount. It says $500 but should be $350.
Agent (Sara): I can see invoice #4521. Let me check the billing records.
Agent (Sara): You're right, there was a duplicate charge. I've submitted a correction.
Customer: When will I see the updated invoice?
Agent (Sara): The corrected invoice should be in your account within 24 hours.
Customer: It's been 2 days and I still see the wrong amount.
Agent (Sara): I'm escalating this to our billing team. They'll follow up within 4 hours.
""")

print(result.summary.issue)
# "Incorrect invoice amount — #4521 shows $500 instead of $350 due to duplicate charge"
print(result.summary.status)
# "escalated"
print(result.summary.next_step)
# "Billing team to follow up and ensure corrected invoice is applied"
```

---

## Example 3: Long document condenser

Summarize long documents that exceed LM context using map-reduce.

### Module

```python
class SummarizeSection(dspy.Signature):
    """Summarize this section of a document, preserving key data and conclusions."""
    section: str = dspy.InputField(desc="A section of a larger document")
    section_summary: str = dspy.OutputField(desc="Key points from this section")

class MergeSummaries(dspy.Signature):
    """Merge section summaries into a coherent executive summary."""
    section_summaries: list[str] = dspy.InputField()
    doc_word_count: int = dspy.InputField(desc="Length of the original document in words")
    executive_summary: str = dspy.OutputField(desc="A unified summary covering all sections")
    key_takeaways: list[str] = dspy.OutputField(desc="3-5 most important takeaways")

class DocumentCondenser(dspy.Module):
    def __init__(self, words_per_chunk=2000):
        self.words_per_chunk = words_per_chunk
        self.summarize_section = dspy.ChainOfThought(SummarizeSection)
        self.merge = dspy.ChainOfThought(MergeSummaries)

    def forward(self, document):
        chunks = self._chunk(document)

        section_summaries = []
        for chunk in chunks:
            result = self.summarize_section(section=chunk)
            section_summaries.append(result.section_summary)

        merged = self.merge(
            section_summaries=section_summaries,
            doc_word_count=len(document.split()),
        )

        dspy.Suggest(
            len(merged.key_takeaways) >= 3,
            "Include at least 3 key takeaways"
        )

        return merged

    def _chunk(self, text):
        words = text.split()
        return [" ".join(words[i:i+self.words_per_chunk])
                for i in range(0, len(words), self.words_per_chunk)]
```

### Usage

```python
condenser = DocumentCondenser(words_per_chunk=2000)

# Works for documents of any length
result = condenser(document=long_report_text)
print(result.executive_summary)
print(result.key_takeaways)
```

### Metric

```python
class JudgeSummaryQuality(dspy.Signature):
    """Judge the quality of a document summary."""
    document_excerpt: str = dspy.InputField(desc="First ~500 words of the original")
    summary: str = dspy.InputField()
    reference_summary: str = dspy.InputField()
    faithfulness: float = dspy.OutputField(desc="0.0-1.0 — no fabricated claims")
    coverage: float = dspy.OutputField(desc="0.0-1.0 — key points captured")
    coherence: float = dspy.OutputField(desc="0.0-1.0 — reads well as standalone text")

def doc_summary_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeSummaryQuality)
    result = judge(
        document_excerpt=example.document[:2000],
        summary=prediction.executive_summary,
        reference_summary=example.executive_summary,
    )
    return (result.faithfulness + result.coverage + result.coherence) / 3
```
