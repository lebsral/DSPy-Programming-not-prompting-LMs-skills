---
name: ai-following-rules
description: Make your AI follow rules and policies. Use when your AI breaks format rules, violates content policies, ignores business constraints, outputs invalid JSON, exceeds length limits, includes forbidden content, or doesn't comply with your specifications. Covers DSPy Assert/Suggest for hard and soft rules, content policies, format enforcement, retry mechanics, and composing multiple constraints.
---

# Make Your AI Follow the Rules

Guide the user through defining and enforcing rules their AI must follow. The key insight: don't ask the AI to follow rules — **program constraints that enforce them automatically**.

## The two types of rules

DSPy gives you two constraint primitives:

| | `dspy.Assert` | `dspy.Suggest` |
|--|--------------|----------------|
| **Behavior** | Hard stop — retries if violated | Soft nudge — continues if violated |
| **Use for** | Must-comply rules (format, safety, legal) | Should-comply preferences (style, tone) |
| **On failure** | LM retries with error feedback | LM gets suggestion, continues |
| **PM translation** | "This **must** happen" | "This **should** happen" |

```python
import dspy

# Hard rule — will retry up to max_backtrack_attempts times
dspy.Assert(
    condition,       # bool: does the output satisfy the rule?
    "error message"  # str: feedback to the LM on what went wrong
)

# Soft rule — nudges but doesn't block
dspy.Suggest(
    condition,
    "suggestion message"
)
```

## Step 1: Identify your rules

Ask the user:
1. **What rules does the AI break?** (too long? wrong format? forbidden content? missing fields?)
2. **Which rules are hard requirements vs nice-to-haves?** (Assert vs Suggest)
3. **What should happen when a rule is broken?** (retry, flag for review, fail loudly)

## Step 2: Content policy rules

Enforce what the AI can and cannot say.

```python
class PolicyCheckedResponse(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        result = self.respond(question=question)
        answer = result.answer

        # Hard rules — must comply
        dspy.Assert(
            len(answer.split()) <= 280,
            f"Response is {len(answer.split())} words. Must be under 280 words."
        )
        dspy.Assert(
            not any(word in answer.lower() for word in BLOCKED_WORDS),
            "Response contains blocked words. Remove them and regenerate."
        )
        dspy.Assert(
            "disclaimer" not in answer.lower(),
            "Do not include disclaimers. Answer directly."
        )

        # Soft rules — prefer but don't block
        dspy.Suggest(
            answer[0].isupper(),
            "Response should start with a capital letter."
        )
        dspy.Suggest(
            answer.endswith(".") or answer.endswith("!") or answer.endswith("?"),
            "Response should end with proper punctuation."
        )

        return result

BLOCKED_WORDS = ["competitor_name", "profanity1", "profanity2"]  # your list
```

## Step 3: Format rules

Enforce output structure — valid JSON, required fields, correct types.

```python
import json
from pydantic import BaseModel, Field
from typing import Literal

# Option A: Pydantic validation (automatic)
class QuizQuestion(BaseModel):
    question: str = Field(min_length=10)
    options: list[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    difficulty: Literal["easy", "medium", "hard"]

class GenerateQuiz(dspy.Signature):
    """Generate a quiz question about the topic."""
    topic: str = dspy.InputField()
    quiz: QuizQuestion = dspy.OutputField()

# Option B: Assert-based validation (custom logic)
class QuizGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateQuiz)

    def forward(self, topic):
        result = self.generate(topic=topic)
        quiz = result.quiz

        # Correct answer must be one of the options
        dspy.Assert(
            quiz.correct_answer in quiz.options,
            f"Correct answer '{quiz.correct_answer}' is not in options {quiz.options}. "
            "The correct answer must be one of the four options."
        )

        # Options must be unique
        dspy.Assert(
            len(set(quiz.options)) == 4,
            "All four options must be different from each other."
        )

        return result
```

Combine Pydantic (catches type/structure errors) with Assert (catches logic errors) for the strongest format enforcement.

## Step 4: Business constraint rules

Translate business requirements into programmatic constraints.

```python
class PricingResponse(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("customer_question, pricing_docs -> answer")

    def forward(self, customer_question, pricing_docs):
        result = self.respond(
            customer_question=customer_question,
            pricing_docs=pricing_docs,
        )

        # Never mention competitor pricing
        dspy.Assert(
            not any(comp in result.answer.lower() for comp in COMPETITORS),
            "Do not mention competitor pricing. Focus only on our plans."
        )

        # Never offer unauthorized discounts
        dspy.Assert(
            "discount" not in result.answer.lower() or "authorized" in result.answer.lower(),
            "Do not offer discounts unless referencing an authorized promotion."
        )

        # Always include a CTA
        dspy.Suggest(
            any(cta in result.answer.lower() for cta in ["contact", "sign up", "learn more", "get started"]),
            "Include a call-to-action at the end of the response."
        )

        return result

COMPETITORS = ["competitor_a", "competitor_b"]
```

## Step 5: How retry and backtracking works

When `dspy.Assert` fails, DSPy doesn't just retry blindly — it feeds the error message back to the LM:

```
Attempt 1: LM generates response → Assert fails ("Response is 350 words, must be under 280")
Attempt 2: LM retries with feedback → Assert fails ("Response contains blocked words")
Attempt 3: LM retries with feedback → Assert passes ✓
```

Key details:
- **Error messages matter.** They're the LM's self-correction instructions. Be specific: "Response is 350 words, must be under 280" is better than "too long."
- **Default retries: 2.** Set via `max_backtrack_attempts` on the module.
- **Each retry sees all previous failures.** The model gets a cumulative error log.
- **Suggest never retries.** It sends the feedback but continues regardless.

## Step 6: Composing multiple rules

Stack rules by putting multiple Assert/Suggest calls in sequence. They're checked in order.

```python
class TweetWriter(dspy.Module):
    def __init__(self):
        self.write = dspy.ChainOfThought("topic, key_facts -> tweet")

    def forward(self, topic, key_facts):
        result = self.write(topic=topic, key_facts=key_facts)
        tweet = result.tweet

        # Rule 1: Length limit (hard)
        dspy.Assert(
            len(tweet) <= 280,
            f"Tweet is {len(tweet)} chars. Must be ≤280."
        )

        # Rule 2: No hashtags (hard)
        dspy.Assert(
            "#" not in tweet,
            "No hashtags allowed. Remove all # symbols."
        )

        # Rule 3: Must include key fact (hard)
        dspy.Assert(
            any(fact.lower() in tweet.lower() for fact in key_facts),
            f"Tweet must mention at least one key fact: {key_facts}"
        )

        # Rule 4: Engaging tone (soft)
        dspy.Suggest(
            not tweet.startswith("Did you know"),
            "Avoid starting with 'Did you know' — be more creative."
        )

        # Rule 5: No emojis (soft)
        dspy.Suggest(
            not any(ord(c) > 127 for c in tweet),
            "Prefer text-only tweets without emojis."
        )

        return result
```

When rules conflict (e.g., "include all key facts" vs "stay under 280 chars"), put the **harder constraint first** so the model prioritizes it.

## Step 7: Optimizing with rules

DSPy optimizers work with assertions. When you optimize a module that has Assert/Suggest:
- The optimizer sees assertion pass/fail rates as part of the metric
- Optimized prompts learn to satisfy constraints more often
- Result: fewer retries needed in production

```python
def metric(example, pred, trace=None):
    # Your quality metric + assertion compliance
    correct = pred.answer == example.expected_answer
    return correct  # assertions are enforced separately during forward()

optimizer = dspy.MIPROv2(metric=metric, num_threads=4)
optimized = optimizer.compile(
    my_module,
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)
```

## Key principles

- **Assert for requirements, Suggest for preferences.** Don't use Assert for style issues.
- **Specific error messages.** "350 words, must be under 280" beats "too long."
- **Pydantic + Assert together.** Pydantic catches structure, Assert catches logic.
- **Order matters.** Put hard constraints before soft ones.
- **Optimize after adding rules.** DSPy learns to comply, reducing runtime retries.

## Additional resources

- Use `/ai-checking-outputs` for general output verification (safety, quality gates)
- Use `/ai-stopping-hallucinations` for grounding AI in facts and sources
- Use `/ai-improving-accuracy` to measure and improve quality after adding rules
- Use `/ai-testing-safety` to verify your rules hold up against adversarial users
- See `examples.md` for complete worked examples
