# ai-rewriting-text - Examples

## Example 1 - Technical-to-plain-English converter

Convert developer documentation into user-friendly help articles.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class TechToPlainEnglish(dspy.Signature):
    """Rewrite technical documentation as a friendly help article for non-technical users.
    Replace all jargon with plain language equivalents.
    Preserve every feature and instruction from the original.
    Do not add new information. Keep output length similar to input length."""

    technical_text: str = dspy.InputField(desc="Developer-facing documentation or technical description")
    user_persona: str = dspy.InputField(desc="Who the help article is for - e.g. 'small business owner', 'first-time user'")
    plain_english_text: str = dspy.OutputField(desc="User-friendly help article version of the documentation")

converter = dspy.Predict(TechToPlainEnglish)

technical_docs = [
    {
        "text": (
            "To authenticate API requests, include your Bearer token in the Authorization header. "
            "Tokens expire after 3600 seconds and must be refreshed using the /oauth/token endpoint "
            "with your client_id and client_secret."
        ),
        "persona": "small business owner with no coding experience",
    },
    {
        "text": (
            "Enable two-factor authentication (2FA) by navigating to Account Settings > Security. "
            "The system supports TOTP-based authenticator apps and SMS fallback. "
            "Recovery codes are generated on initial 2FA setup."
        ),
        "persona": "general user setting up their account",
    },
]

for doc in technical_docs:
    result = converter(
        technical_text=doc["text"],
        user_persona=doc["persona"],
    )
    print(f"Original:\n{doc['text']}\n")
    print(f"Plain English:\n{result.plain_english_text}\n")
    print("---")
```

**What to expect:**
- "Bearer token in the Authorization header" → "a password that proves who you are"
- "TOTP-based authenticator apps" → "an app on your phone that shows a code"
- Output length stays proportional to input

---

## Example 2 - Tone adapter (formal report to casual blog post)

Transform a formal quarterly business report excerpt into a casual, readable blog post.

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class ToneAdapter(dspy.Signature):
    """Rewrite the source text in the target tone while preserving all facts and figures.
    Match the energy and vocabulary of the style examples provided.
    Do not add new claims or omit any data points from the original.
    Output length should be similar to input length."""

    source_text: str = dspy.InputField(desc="Formal text to rewrite")
    target_tone: str = dspy.InputField(desc="Description of target tone")
    style_examples: str = dspy.InputField(desc="2-3 example passages in the target style, separated by ---")
    rewritten_text: str = dspy.OutputField(desc="Rewritten text in the target tone")

# Style examples of casual, conversational blog writing
casual_style_examples = """
Q3 was a good one. Revenue climbed 18% and we finally cracked the enterprise market we'd been eyeing.
---
Here's the short version - we shipped faster, customers stuck around longer, and the team grew without breaking anything.
---
The numbers tell a clear story this quarter. Churn dropped. Signups jumped. The product finally clicked for users.
"""

formal_passages = [
    (
        "Revenue for the fiscal quarter increased by 23% year-over-year, reaching $4.2M. "
        "This growth was primarily attributable to expansion within the mid-market segment "
        "and a 15% improvement in net revenue retention."
    ),
    (
        "Customer acquisition cost (CAC) decreased by 12% relative to the prior period, "
        "driven by optimization of paid acquisition channels and increased organic referral volume. "
        "Payback period improved from 14 months to 11 months."
    ),
]

adapter = dspy.Predict(ToneAdapter)

for passage in formal_passages:
    result = adapter(
        source_text=passage,
        target_tone="casual, conversational, founder blog post",
        style_examples=casual_style_examples,
    )
    print(f"Formal:\n{passage}\n")
    print(f"Casual:\n{result.rewritten_text}\n")
    print("---")
```

### Adding a fidelity check

```python
class FidelityJudge(dspy.Signature):
    """Score how well the rewritten text preserves all numbers, percentages, and factual claims
    from the original. Score 0-1 where 1 means every data point is present and accurate."""

    source_text: str = dspy.InputField()
    rewritten_text: str = dspy.InputField()
    fidelity_score: float = dspy.OutputField(desc="Float 0-1")
    missing_or_changed: str = dspy.OutputField(desc="List any missing or altered facts, or 'none'")

judge = dspy.Predict(FidelityJudge)

for passage in formal_passages:
    result = adapter(
        source_text=passage,
        target_tone="casual, conversational, founder blog post",
        style_examples=casual_style_examples,
    )
    check = judge(source_text=passage, rewritten_text=result.rewritten_text)
    print(f"Fidelity score - {check.fidelity_score}")
    print(f"Issues - {check.missing_or_changed}\n")
```

---

## Example 3 - Reading level adjuster with measurement

Adjust a college-level passage to 8th grade reading level and verify with textstat.

```python
import dspy

# pip install textstat
import textstat

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class ReadingLevelAdjuster(dspy.Signature):
    """Rewrite the source text to match the target reading level.
    Use vocabulary, sentence length, and structure appropriate for the grade level.
    Preserve all key information and concepts from the original.
    Do not add new information. Keep output length similar to input length."""

    source_text: str = dspy.InputField(desc="Original text at its current reading level")
    target_reading_level: str = dspy.InputField(desc="Target grade level - e.g. '8th grade', '5th grade', 'college'")
    rewritten_text: str = dspy.OutputField(desc="Text rewritten at the target reading level")

adjuster = dspy.Predict(ReadingLevelAdjuster)

college_level_passages = [
    (
        "The mitochondria, often referred to as the powerhouse of the cell, are double-membrane-bound "
        "organelles responsible for the production of adenosine triphosphate (ATP) through oxidative "
        "phosphorylation, a process integral to cellular respiration and energy metabolism."
    ),
    (
        "Macroeconomic policy instruments encompass fiscal and monetary mechanisms deployed by governmental "
        "and central banking authorities to regulate aggregate demand, mitigate inflationary pressures, "
        "and stabilize employment levels across economic cycles."
    ),
]

target_level = "8th grade"

for passage in college_level_passages:
    original_grade = textstat.flesch_kincaid_grade(passage)

    result = adjuster(
        source_text=passage,
        target_reading_level=target_level,
    )

    rewritten_grade = textstat.flesch_kincaid_grade(result.rewritten_text)

    print(f"Original (grade {original_grade:.1f}):\n{passage}\n")
    print(f"Rewritten (grade {rewritten_grade:.1f}):\n{result.rewritten_text}\n")
    print("---")
```

### Iterative adjustment with retry

If the first rewrite misses the target grade level, retry with explicit feedback.

```python
def adjust_to_grade(source_text: str, target_grade: float, max_attempts: int = 3) -> str:
    """Rewrite text to target grade level, retrying if the measured level is off."""
    current_text = source_text
    current_instruction = f"{target_grade:.0f}th grade"

    for attempt in range(max_attempts):
        result = adjuster(
            source_text=source_text,  # always rewrite from original
            target_reading_level=current_instruction,
        )
        measured = textstat.flesch_kincaid_grade(result.rewritten_text)

        if abs(measured - target_grade) <= 1.5:
            print(f"Hit target on attempt {attempt + 1} (measured grade {measured:.1f})")
            return result.rewritten_text

        # Give corrective feedback for the next attempt
        if measured > target_grade:
            current_instruction = f"{target_grade:.0f}th grade - use shorter sentences and simpler words (current text is grade {measured:.1f}, too complex)"
        else:
            current_instruction = f"{target_grade:.0f}th grade - you can use slightly more sophisticated vocabulary (current text is grade {measured:.1f}, too simple)"

    return result.rewritten_text  # return best attempt

adjusted = adjust_to_grade(college_level_passages[0], target_grade=8.0)
print(adjusted)
```
