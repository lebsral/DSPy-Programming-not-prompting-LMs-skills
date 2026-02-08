---
name: ai-checking-outputs
description: Verify and validate AI output before it reaches users. Use when you need guardrails, output validation, safety checks, content filtering, fact-checking AI responses, catching hallucinations, preventing bad outputs, quality gates, or ensuring AI responses meet your standards before shipping them. Covers DSPy assertions, verification patterns, and generate-then-filter pipelines.
---

# Check AI Output Before It Ships

Guide the user through adding verification and guardrails so bad AI outputs never reach users. The pattern: generate, check, fix or reject.

## Step 1: Understand what to check

Ask the user:
1. **What could go wrong?** (hallucinations, wrong format, offensive content, missing info, factual errors?)
2. **How strict does it need to be?** (reject bad outputs vs. try to fix them?)
3. **What's the cost of a bad output reaching users?** (annoyance vs. legal/safety risk)

## Step 2: Quick wins — DSPy assertions

The simplest way to add checks. `dspy.Assert` is a hard stop (retry if violated), `dspy.Suggest` is a soft nudge:

```python
import dspy

class CheckedResponder(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought(GenerateResponse)

    def forward(self, question):
        result = self.respond(question=question)

        # Hard checks — will retry if these fail
        dspy.Assert(
            len(result.answer) > 0,
            "Must produce an answer"
        )
        dspy.Assert(
            len(result.answer.split()) <= 200,
            "Answer must be under 200 words"
        )

        # Soft checks — hints for improvement
        dspy.Suggest(
            "I don't know" not in result.answer.lower(),
            "Try to provide a substantive answer"
        )
        dspy.Suggest(
            not any(word in result.answer.lower() for word in ["definitely", "absolutely", "100%"]),
            "Avoid overconfident language"
        )

        return result
```

DSPy will automatically retry the LM call (with the assertion feedback) when an `Assert` fails, up to a configurable number of times.

## Step 3: Format validation

### Type-based validation (automatic)

DSPy validates typed outputs automatically:

```python
from typing import Literal
from pydantic import BaseModel, Field

class Response(BaseModel):
    answer: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)
    category: str

class MySignature(dspy.Signature):
    question: str = dspy.InputField()
    response: Response = dspy.OutputField()
```

Pydantic catches malformed JSON, out-of-range values, and wrong types before your code ever sees them.

### Custom validation in the module

```python
import re

class ValidatedExtractor(dspy.Module):
    def __init__(self):
        self.extract = dspy.ChainOfThought(ExtractContact)

    def forward(self, text):
        result = self.extract(text=text)

        # Validate email format
        dspy.Assert(
            re.match(r"[^@]+@[^@]+\.[^@]+", result.email or ""),
            "Email must be a valid email address"
        )

        # Validate phone format
        dspy.Assert(
            len(re.sub(r"\D", "", result.phone or "")) >= 10,
            "Phone must have at least 10 digits"
        )

        return result
```

## Step 4: Factual verification

### Self-check — ask the AI to verify its own output

```python
class VerifyFacts(dspy.Signature):
    """Check if the answer is supported by the given context."""
    context: list[str] = dspy.InputField(desc="Source documents")
    answer: str = dspy.InputField(desc="Generated answer to verify")
    is_supported: bool = dspy.OutputField(desc="Is the answer fully supported by the context?")
    unsupported_claims: list[str] = dspy.OutputField(desc="Claims not found in context")

class GroundedResponder(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.answer = dspy.ChainOfThought(AnswerFromDocs)
        self.verify = dspy.Predict(VerifyFacts)

    def forward(self, question):
        context = self.retrieve(question).passages
        response = self.answer(context=context, question=question)

        # Verify the answer is grounded in sources
        check = self.verify(context=context, answer=response.answer)
        dspy.Assert(
            check.is_supported,
            f"Answer contains unsupported claims: {check.unsupported_claims}. "
            "Rewrite using only information from the context."
        )

        return response
```

### Cross-check — generate two ways, compare

```python
class CrossCheckedAnswer(dspy.Module):
    def __init__(self):
        self.answer_a = dspy.ChainOfThought(AnswerQuestion)
        self.answer_b = dspy.ChainOfThought(AnswerQuestion)
        self.compare = dspy.ChainOfThought(CompareAnswers)

    def forward(self, question):
        a = self.answer_a(question=question)
        b = self.answer_b(question=question)

        comparison = self.compare(
            question=question,
            answer_a=a.answer,
            answer_b=b.answer,
        )

        dspy.Assert(
            comparison.agree,
            "Two independent generations disagree — the answer may be unreliable"
        )

        return a

class CompareAnswers(dspy.Signature):
    """Check if two independently generated answers agree."""
    question: str = dspy.InputField()
    answer_a: str = dspy.InputField()
    answer_b: str = dspy.InputField()
    agree: bool = dspy.OutputField(desc="Do the answers substantially agree?")
    discrepancy: str = dspy.OutputField(desc="What they disagree on, if anything")
```

## Step 5: Safety and content filtering

### Block harmful outputs

```python
BLOCKED_PATTERNS = [
    r"\b(password|secret|api.?key)\b",
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
]

class SafeResponder(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought(GenerateResponse)

    def forward(self, question):
        result = self.respond(question=question)

        # Check for leaked sensitive data
        for pattern in BLOCKED_PATTERNS:
            dspy.Assert(
                not re.search(pattern, result.answer, re.IGNORECASE),
                f"Response may contain sensitive data (pattern: {pattern})"
            )

        return result
```

### AI-as-safety-judge

```python
class SafetyCheck(dspy.Signature):
    """Check if the response is safe and appropriate."""
    question: str = dspy.InputField()
    response: str = dspy.InputField()
    is_safe: bool = dspy.OutputField()
    concern: str = dspy.OutputField(desc="Safety concern if not safe, empty if safe")

class SafetyCheckedResponder(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought(GenerateResponse)
        self.check = dspy.Predict(SafetyCheck)

    def forward(self, question):
        result = self.respond(question=question)

        safety = self.check(question=question, response=result.answer)
        dspy.Assert(
            safety.is_safe,
            f"Response flagged as unsafe: {safety.concern}. Regenerate."
        )

        return result
```

## Step 6: Generate → Filter → Pick best (ensemble pattern)

For high-stakes outputs, generate multiple candidates and filter:

```python
class FilteredEnsemble(dspy.Module):
    def __init__(self, num_candidates=5):
        self.generators = [dspy.ChainOfThought(GenerateAnswer) for _ in range(num_candidates)]
        self.judge = dspy.ChainOfThought(RankAnswers)

    def forward(self, question):
        candidates = []
        for gen in self.generators:
            try:
                result = gen(question=question)
                # Only keep candidates that pass basic checks
                if len(result.answer) > 0 and len(result.answer.split()) < 200:
                    candidates.append(result.answer)
            except Exception:
                continue

        dspy.Assert(len(candidates) > 0, "No valid candidates generated")

        return self.judge(question=question, candidates=candidates)

class RankAnswers(dspy.Signature):
    """Pick the best answer from the candidates."""
    question: str = dspy.InputField()
    candidates: list[str] = dspy.InputField()
    best_answer: str = dspy.OutputField()
```

## Key patterns

- **Assert for hard requirements** — format, length, safety. DSPy retries automatically.
- **Suggest for soft preferences** — style, tone, detail level. Won't block but nudges.
- **Pydantic for structure** — catches malformed output automatically.
- **Self-verification for facts** — ask the AI "is this grounded in the sources?"
- **Cross-checking for reliability** — generate twice independently, compare.
- **Regex for sensitive data** — block SSNs, API keys, passwords in output.
- **Ensemble for high stakes** — generate many, filter, pick the best.

## Checklist: what to check

| Check | When to use | How |
|-------|------------|-----|
| Non-empty output | Always | `dspy.Assert(len(answer) > 0, ...)` |
| Length limits | User-facing text | `dspy.Assert(len(answer.split()) < N, ...)` |
| Valid format | Structured output | Pydantic model + `dspy.Assert` |
| Grounded in sources | RAG / doc search | Verification signature |
| No sensitive data | Any user-facing output | Regex patterns |
| Safe content | Public-facing apps | AI safety judge |
| Consistent | Critical decisions | Cross-check with two generations |
| High quality | High-stakes outputs | Ensemble + ranking |

## Additional resources

- Use `/ai-building-pipelines` to wire checks into multi-step systems
- Use `/ai-making-consistent` for output consistency (not correctness)
- Next: `/ai-improving-accuracy` to measure and improve quality
