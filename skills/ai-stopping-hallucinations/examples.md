# Anti-Hallucination Examples

## Example 1: Customer support grounded in help docs

A support chatbot that only answers from your help center — refuses to speculate.

```python
import dspy
import re

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class CitedSupportAnswer(dspy.Signature):
    """Answer a customer question using only the help docs. Cite every claim with [1], [2], etc.
    If the docs don't cover the question, say so — don't guess."""
    help_docs: list[str] = dspy.InputField(desc="Numbered help center passages")
    question: str = dspy.InputField(desc="Customer's question")
    answer: str = dspy.OutputField(desc="Answer citing help docs, or 'I don't have info on that'")

class CheckSupported(dspy.Signature):
    """Verify every claim in the answer appears in the help docs."""
    help_docs: list[str] = dspy.InputField()
    answer: str = dspy.InputField()
    is_supported: bool = dspy.OutputField()
    unsupported_claims: list[str] = dspy.OutputField(desc="Claims not in the docs")

class GroundedSupportBot(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought(CitedSupportAnswer)
        self.verify = dspy.Predict(CheckSupported)

    def forward(self, help_docs, question):
        result = self.answer(help_docs=help_docs, question=question)

        # Enforce citations exist
        sentences = [s.strip() for s in result.answer.split(".") if s.strip()]
        cited = [bool(re.search(r"\[\d+\]", s)) for s in sentences]
        coverage = sum(cited) / max(len(sentences), 1)
        dspy.Assert(
            coverage >= 0.5 or "don't have info" in result.answer.lower(),
            f"Only {coverage:.0%} of sentences cite sources. "
            "Cite with [1], [2], or say you don't have info."
        )

        # Verify faithfulness
        check = self.verify(help_docs=help_docs, answer=result.answer)
        dspy.Assert(
            check.is_supported,
            f"Unsupported claims: {check.unsupported_claims}. "
            "Only state what the help docs say."
        )

        return result

# Usage
docs = [
    "[1] Free plan includes 1,000 API calls per month.",
    "[2] Pro plan is $29/month with 50,000 API calls.",
    "[3] Refunds are available within 14 days of purchase.",
    "[4] To cancel, go to Settings > Billing > Cancel Plan.",
]

bot = GroundedSupportBot()
result = bot(help_docs=docs, question="How do I cancel and get a refund?")
print(result.answer)
# "To cancel your plan, go to Settings > Billing > Cancel Plan [4].
#  Refunds are available within 14 days of purchase [3]."
```

## Example 2: Meeting transcript fact-checker

Verify claims made in a meeting summary against the actual transcript.

```python
import dspy
import re

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# --- Load transcript ---

def load_vtt(path):
    text = open(path).read()
    lines = [line.strip() for line in text.split("\n")
             if line.strip() and not line.startswith("WEBVTT")
             and not re.match(r"\d{2}:\d{2}", line)
             and not line.strip().isdigit()]
    return " ".join(lines)

def chunk_transcript(text, max_chars=400):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > max_chars and current:
            chunks.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current.strip())
    return [f"[{i+1}] {c}" for i, c in enumerate(chunks)]

# --- Signatures ---

class VerifyClaim(dspy.Signature):
    """Check if a specific claim is supported by the transcript."""
    transcript_passages: list[str] = dspy.InputField()
    claim: str = dspy.InputField()
    is_supported: bool = dspy.OutputField()
    supporting_passage: str = dspy.OutputField(desc="Which passage supports it, or 'none'")

class ExtractClaims(dspy.Signature):
    """Extract individual factual claims from a meeting summary."""
    summary: str = dspy.InputField()
    claims: list[str] = dspy.OutputField(desc="List of individual factual claims")

# --- Module ---

class TranscriptFactChecker(dspy.Module):
    def __init__(self):
        self.extract = dspy.Predict(ExtractClaims)
        self.verify = dspy.ChainOfThought(VerifyClaim)

    def forward(self, transcript_passages, summary):
        extracted = self.extract(summary=summary)

        results = []
        for claim in extracted.claims:
            check = self.verify(
                transcript_passages=transcript_passages,
                claim=claim,
            )
            results.append({
                "claim": claim,
                "supported": check.is_supported,
                "source": check.supporting_passage,
            })

        supported = sum(1 for r in results if r["supported"])
        total = len(results)

        return dspy.Prediction(
            claims=results,
            supported_count=supported,
            total_claims=total,
            accuracy=supported / max(total, 1),
        )

# Usage
transcript_text = load_vtt("standup.vtt")
passages = chunk_transcript(transcript_text)

checker = TranscriptFactChecker()
result = checker(
    transcript_passages=passages,
    summary="The team decided to postpone the launch to April. "
            "Sarah will handle the migration. Budget was approved at $75k."
)

for claim in result.claims:
    status = "supported" if claim["supported"] else "NOT SUPPORTED"
    print(f"  [{status}] {claim['claim']}")
print(f"\nAccuracy: {result.accuracy:.0%}")
```

## Example 3: Medical Q&A with layered verification

High-stakes domain: combine citation enforcement + faithfulness verification + confidence gating.

```python
import dspy
import re

lm = dspy.LM("openai/gpt-4o")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class MedicalAnswer(dspy.Signature):
    """Answer a medical question using only the provided clinical guidelines.
    Cite every claim. If unsure, say so explicitly."""
    guidelines: list[str] = dspy.InputField(desc="Numbered clinical guideline passages")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Cited answer from guidelines only")
    confidence: float = dspy.OutputField(desc="0.0-1.0 confidence based on guideline coverage")

class VerifyMedicalClaim(dspy.Signature):
    """Strictly verify: is every medical claim in the answer directly stated in the guidelines?"""
    guidelines: list[str] = dspy.InputField()
    answer: str = dspy.InputField()
    is_faithful: bool = dspy.OutputField()
    unsupported_claims: list[str] = dspy.OutputField()
    risk_level: str = dspy.OutputField(desc="low/medium/high — risk if claim is wrong")

class SafeMedicalQA(dspy.Module):
    def __init__(self, confidence_threshold=0.8):
        self.answer = dspy.ChainOfThought(MedicalAnswer)
        self.verify = dspy.Predict(VerifyMedicalClaim)
        self.threshold = confidence_threshold

        # Use cheaper model for verification
        self.verify.set_lm(dspy.LM("openai/gpt-4o-mini"))  # cheaper model for verification

    def forward(self, guidelines, question):
        result = self.answer(guidelines=guidelines, question=question)

        # Layer 1: Citation enforcement
        cited_nums = set(int(n) for n in re.findall(r"\[(\d+)\]", result.answer))
        valid_nums = set(range(1, len(guidelines) + 1))
        dspy.Assert(
            cited_nums.issubset(valid_nums),
            f"Invalid citations: {cited_nums - valid_nums}. "
            f"Only use [1] to [{len(guidelines)}]."
        )

        # Layer 2: Faithfulness verification
        check = self.verify(guidelines=guidelines, answer=result.answer)
        dspy.Assert(
            check.is_faithful,
            f"MEDICAL SAFETY: Unsupported claims detected: {check.unsupported_claims}. "
            "In medical contexts, only state what the guidelines explicitly say."
        )

        # Layer 3: Confidence gating
        needs_review = (
            result.confidence < self.threshold
            or check.risk_level == "high"
        )

        return dspy.Prediction(
            answer=result.answer,
            confidence=result.confidence,
            needs_review=needs_review,
            risk_level=check.risk_level,
        )

# Usage
qa = SafeMedicalQA(confidence_threshold=0.8)
guidelines = [
    "[1] Standard adult ibuprofen dose: 200-400mg every 4-6 hours.",
    "[2] Maximum daily dose without supervision: 1200mg.",
    "[3] Common side effects: stomach upset, dizziness, headache.",
    "[4] Contraindicated with blood thinners and in renal impairment.",
]
result = qa(guidelines=guidelines, question="Can I take ibuprofen with aspirin?")
if result.needs_review:
    print(f"NEEDS PHYSICIAN REVIEW (confidence: {result.confidence}, risk: {result.risk_level})")
print(result.answer)
```
