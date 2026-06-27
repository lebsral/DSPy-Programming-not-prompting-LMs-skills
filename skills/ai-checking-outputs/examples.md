# Checking Outputs Examples

## Format Validation — Pydantic (Zero-Code Guardrail)

When you need type, range, or structure validation, DSPy's typed signatures handle this automatically through Pydantic — no reward function needed.

```python
import dspy
from pydantic import BaseModel, Field
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5", etc.
dspy.configure(lm=lm)

class ProductSummary(BaseModel):
    headline: str = Field(min_length=10, max_length=120)
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)

class SummarizeReview(dspy.Signature):
    """Summarize the product review."""
    review: str = dspy.InputField()
    summary: ProductSummary = dspy.OutputField()

summarizer = dspy.ChainOfThought(SummarizeReview)
result = summarizer(review="Great build quality but the battery drains too fast.")
print(result.summary.headline)    # "Solid build quality undermined by poor battery life"
print(result.summary.sentiment)   # "negative"
print(result.summary.confidence)  # 0.82
```

Pydantic catches malformed JSON, out-of-range values, and enum violations automatically. Upgrade to `dspy.Refine` only when Pydantic cannot express the constraint (e.g., logic rules that span multiple fields).

## Format Validation — Logic Rules with dspy.Refine

When Pydantic is not enough, express the constraint as a reward function and wrap the module with `dspy.Refine`.

```python
import dspy
import re

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5", etc.
dspy.configure(lm=lm)

class ExtractContact(dspy.Signature):
    text: str = dspy.InputField()
    email: str = dspy.OutputField()
    phone: str = dspy.OutputField()

class ContactExtractor(dspy.Module):
    def __init__(self):
        self.extract = dspy.ChainOfThought(ExtractContact)

    def forward(self, text):
        return self.extract(text=text)

def contact_format_reward(args: dict, pred: dspy.Prediction) -> float:
    email_ok = bool(re.match(r"[^@]+@[^@]+\.[^@]+", pred.email or ""))
    phone_digits = len(re.sub(r"\D", "", pred.phone or ""))
    if not email_ok or phone_digits < 10:
        return 0.0
    return 1.0

validated = dspy.Refine(ContactExtractor(), N=3, reward_fn=contact_format_reward, threshold=1.0)
result = validated(text="Reach us at sales@acme.com or call 415-555-0199.")
print(result.email)  # "sales@acme.com"
print(result.phone)  # "415-555-0199"
```

Use `threshold=1.0` for binary pass/fail checks — only a perfect score passes. Use a lower threshold (e.g., `0.8`) when the reward function returns partial scores.

## Fact-Checking Gate

Use a verification signature inside the reward function to confirm generated answers are grounded in source documents.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5", etc.
dspy.configure(lm=lm)

class AnswerFromDocs(dspy.Signature):
    """Answer the question using only the provided documents."""
    context: list[str] = dspy.InputField(desc="Source documents")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class VerifyGrounding(dspy.Signature):
    """Check if the answer is fully supported by the given context."""
    context: list[str] = dspy.InputField(desc="Source documents")
    answer: str = dspy.InputField()
    is_supported: bool = dspy.OutputField()
    unsupported_claims: list[str] = dspy.OutputField(desc="Claims not found in context")

# Instantiate the verifier ONCE outside the reward function — never inside it
verifier = dspy.Predict(VerifyGrounding)

class GroundedAnswerer(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought(AnswerFromDocs)

    def forward(self, context, question):
        return self.answer(context=context, question=question)

def faithfulness_reward(args: dict, pred: dspy.Prediction) -> float:
    check = verifier(context=args["context"], answer=pred.answer)
    return 1.0 if check.is_supported else 0.0

grounded = dspy.Refine(GroundedAnswerer(), N=3, reward_fn=faithfulness_reward, threshold=1.0)
result = grounded(
    context=["Q3 revenue was $4.2M, up 18% year-over-year."],
    question="What was Q3 revenue?",
)
print(result.answer)  # "Q3 revenue was $4.2M, an 18% increase year-over-year."
```

AI-as-verifier doubles LM calls per attempt. Reserve it for high-stakes outputs; use regex or Pydantic for lower-stakes checks.

## Safety Filter

Block sensitive data and harmful outputs before they reach users.

```python
import dspy
import re

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5", etc.
dspy.configure(lm=lm)

SENSITIVE_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",           # SSN
    r"\b(?:password|api.?key|secret)\b", # Credential keywords
]

class SafetyCheck(dspy.Signature):
    """Check whether the response is safe and appropriate for users."""
    question: str = dspy.InputField()
    response: str = dspy.InputField()
    is_safe: bool = dspy.OutputField()
    concern: str = dspy.OutputField(desc="What is unsafe, or empty string if safe")

# Instantiate the safety judge once, not inside the reward function
safety_judge = dspy.Predict(SafetyCheck)

class SupportResponder(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.respond(question=question)

def safety_reward(args: dict, pred: dspy.Prediction) -> float:
    answer = pred.answer or ""
    # Regex first — cheap, no LM call
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            return 0.0
    # AI judge for nuanced checks
    check = safety_judge(question=args["question"], response=answer)
    return 1.0 if check.is_safe else 0.0

safe_responder = dspy.Refine(SupportResponder(), N=3, reward_fn=safety_reward, threshold=1.0)
result = safe_responder(question="How do I reset my account?")
print(result.answer)  # "Visit account.example.com/reset to set a new password."
```

Run regex checks first — they're fast and catch obvious issues before spending an LM call on the AI judge.

## End-to-End Production Pattern: Support Bot with Guardrails

A realistic support bot combining Pydantic structure validation, sensitive data filtering, and faithfulness checking in one reward function.

```python
import dspy
import re
from pydantic import BaseModel, Field

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5", etc.
dspy.configure(lm=lm)

# ── Structured output ─────────────────────────────────────────────────────────
class SupportReply(BaseModel):
    message: str = Field(min_length=10, max_length=500)
    suggested_action: str = Field(description="Next step for the user")
    escalate: bool = Field(description="Should this go to a human agent?")

class GenerateSupportReply(dspy.Signature):
    """Generate a helpful support reply grounded in the provided knowledge base."""
    kb_articles: list[str] = dspy.InputField(desc="Relevant knowledge base articles")
    customer_message: str = dspy.InputField()
    reply: SupportReply = dspy.OutputField()

# ── Faithfulness verifier — instantiated once at module level ─────────────────
class CheckGrounding(dspy.Signature):
    """Is the support reply grounded in the knowledge base articles?"""
    kb_articles: list[str] = dspy.InputField()
    reply_message: str = dspy.InputField()
    is_grounded: bool = dspy.OutputField()

grounding_check = dspy.Predict(CheckGrounding)

SENSITIVE = [r"\b\d{3}-\d{2}-\d{4}\b", r"\b(?:password|api.?key)\b"]

# ── Combined reward function ───────────────────────────────────────────────────
def guardrail_reward(args: dict, pred: dspy.Prediction) -> float:
    reply = pred.reply
    text = reply.message
    score = 1.0

    # Hard: block sensitive data (fast, no LM call)
    for pat in SENSITIVE:
        if re.search(pat, text, re.IGNORECASE):
            return 0.0

    # Hard: must be grounded in the knowledge base
    check = grounding_check(kb_articles=args["kb_articles"], reply_message=text)
    if not check.is_grounded:
        return 0.0

    # Soft: prefer replies with a clear action
    if len(reply.suggested_action.strip()) < 5:
        score -= 0.2

    return score

# ── Module + wrapped production version ───────────────────────────────────────
class SupportBot(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateSupportReply)

    def forward(self, kb_articles, customer_message):
        return self.generate(kb_articles=kb_articles, customer_message=customer_message)

production_bot = dspy.Refine(
    SupportBot(),
    N=3,
    reward_fn=guardrail_reward,
    threshold=0.8,
)

# ── Usage ──────────────────────────────────────────────────────────────────────
kb = [
    "Password resets are available at account.example.com/reset.",
    "Standard tier SLA is 24 hours. Premium tier SLA is 4 hours.",
]
result = production_bot(
    kb_articles=kb,
    customer_message="I have been waiting 3 days for a response!",
)
print(result.reply.message)
# "We're sorry for the delay — standard SLA is 24 hours and we'll escalate
#  this to a senior agent immediately."
print(result.reply.suggested_action)  # "Escalate to senior support agent"
print(result.reply.escalate)          # True
```

This pattern combines:
- Pydantic for structural guarantees (correct types, length limits, required fields)
- Regex before AI calls to filter obvious violations cheaply
- AI grounding check for faithfulness to the knowledge base
- Soft scoring for preferred-but-not-required qualities (`threshold=0.8`)

To reduce retries in production, optimize the base module with your reward function as the metric first — see `/ai-improving-accuracy`. The model learns to satisfy guardrails on the first attempt, cutting retry cost significantly.
