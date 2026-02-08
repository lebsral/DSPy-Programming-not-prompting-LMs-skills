# Following Rules — Examples

## Content Policy Enforcement

A customer-facing chatbot that must follow brand voice guidelines.

```python
import dspy
import re

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

BRAND_RULES = {
    "blocked_phrases": ["to be honest", "actually", "no offense"],
    "required_sign_off": "— The Acme Team",
    "max_words": 150,
    "tone": "friendly and professional",
}

class BrandResponse(dspy.Signature):
    """Respond to the customer in a friendly and professional tone."""
    customer_message: str = dspy.InputField()
    context: str = dspy.InputField(desc="Relevant knowledge base info")
    response: str = dspy.OutputField()

class BrandCompliantBot(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought(BrandResponse)

    def forward(self, customer_message, context):
        result = self.respond(customer_message=customer_message, context=context)
        response = result.response

        # Hard: word limit
        word_count = len(response.split())
        dspy.Assert(
            word_count <= BRAND_RULES["max_words"],
            f"Response is {word_count} words. Keep it under {BRAND_RULES['max_words']}."
        )

        # Hard: no blocked phrases
        for phrase in BRAND_RULES["blocked_phrases"]:
            dspy.Assert(
                phrase.lower() not in response.lower(),
                f"Remove the phrase '{phrase}' — it's not part of our brand voice."
            )

        # Hard: no competitor mentions
        dspy.Assert(
            "competitor" not in response.lower(),
            "Do not mention competitors. Focus on our product."
        )

        # Soft: include sign-off
        dspy.Suggest(
            response.strip().endswith(BRAND_RULES["required_sign_off"]),
            f"End with '{BRAND_RULES['required_sign_off']}'"
        )

        return result

# Usage
bot = BrandCompliantBot()
result = bot(
    customer_message="Why should I use your product instead of CompetitorX?",
    context="Acme offers 24/7 support, 99.9% uptime, and a free tier.",
)
print(result.response)
```

## JSON Format Enforcement

An API that generates quiz questions — must output valid, logically consistent JSON.

```python
from pydantic import BaseModel, Field
from typing import Literal

class QuizQuestion(BaseModel):
    question: str = Field(min_length=10, description="The quiz question")
    options: list[str] = Field(min_length=4, max_length=4, description="Four answer choices")
    correct_answer: str = Field(description="Must be one of the options")
    explanation: str = Field(min_length=10, description="Why this is correct")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Difficulty level")

class GenerateQuiz(dspy.Signature):
    """Generate a multiple-choice quiz question about the given topic."""
    topic: str = dspy.InputField()
    difficulty: str = dspy.InputField(desc="easy, medium, or hard")
    quiz: QuizQuestion = dspy.OutputField()

class ValidatedQuizGen(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateQuiz)

    def forward(self, topic, difficulty="medium"):
        result = self.generate(topic=topic, difficulty=difficulty)
        quiz = result.quiz

        # Pydantic already validated types and lengths.
        # Now check logic constraints:

        # Correct answer must be in options
        dspy.Assert(
            quiz.correct_answer in quiz.options,
            f"correct_answer '{quiz.correct_answer}' must be one of options: {quiz.options}"
        )

        # All options must be unique
        dspy.Assert(
            len(set(quiz.options)) == len(quiz.options),
            f"Options must be unique. Got duplicates in: {quiz.options}"
        )

        # Explanation should mention the correct answer
        dspy.Suggest(
            quiz.correct_answer.lower() in quiz.explanation.lower(),
            "Explanation should reference the correct answer."
        )

        return result

# Usage
gen = ValidatedQuizGen()
result = gen(topic="Python programming", difficulty="medium")
print(result.quiz.model_dump_json(indent=2))
```

## Business Constraint Enforcement

A pricing chatbot that must follow sales rules.

```python
AUTHORIZED_DISCOUNTS = {
    "WELCOME10": 0.10,
    "ANNUAL20": 0.20,
}

PRICING = {
    "starter": 29,
    "professional": 99,
    "enterprise": "custom",
}

class PricingAnswer(dspy.Signature):
    """Answer the pricing question using our official pricing."""
    question: str = dspy.InputField()
    pricing_info: str = dspy.InputField(desc="Official pricing data")
    answer: str = dspy.OutputField()

class PricingBot(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought(PricingAnswer)

    def forward(self, question):
        pricing_info = (
            f"Plans: Starter ${PRICING['starter']}/mo, "
            f"Professional ${PRICING['professional']}/mo, "
            f"Enterprise: contact sales. "
            f"Active promotions: {', '.join(AUTHORIZED_DISCOUNTS.keys())}"
        )
        result = self.respond(question=question, pricing_info=pricing_info)

        # Never invent prices
        import re
        dollar_amounts = re.findall(r"\$(\d+)", result.answer)
        valid_prices = {str(v) for v in PRICING.values() if isinstance(v, int)}
        for amount in dollar_amounts:
            dspy.Assert(
                amount in valid_prices,
                f"${amount} is not an official price. Valid prices: {valid_prices}"
            )

        # Never offer unauthorized discounts
        discount_mentions = re.findall(r"(\d+)%\s*(?:off|discount)", result.answer.lower())
        authorized_percents = {str(int(v * 100)) for v in AUTHORIZED_DISCOUNTS.values()}
        for pct in discount_mentions:
            dspy.Assert(
                pct in authorized_percents,
                f"{pct}% discount is not authorized. Valid discounts: {AUTHORIZED_DISCOUNTS}"
            )

        # Always suggest contacting sales for enterprise
        if "enterprise" in question.lower():
            dspy.Suggest(
                "contact" in result.answer.lower() or "sales" in result.answer.lower(),
                "For enterprise questions, always suggest contacting sales."
            )

        return result

# Usage
bot = PricingBot()
result = bot(question="Can I get a discount on the Professional plan?")
print(result.answer)
```

## Compliance Logging

Wrap any rule-following module to log assertion pass/fail rates for auditing.

```python
import time
from dataclasses import dataclass, field

@dataclass
class ComplianceLog:
    """Track assertion pass/fail rates for compliance reporting."""
    entries: list[dict] = field(default_factory=list)

    def log(self, rule_name: str, passed: bool, details: str = ""):
        self.entries.append({
            "timestamp": time.time(),
            "rule": rule_name,
            "passed": passed,
            "details": details,
        })

    def pass_rate(self, rule_name: str = None):
        relevant = self.entries
        if rule_name:
            relevant = [e for e in relevant if e["rule"] == rule_name]
        if not relevant:
            return 0.0
        return sum(e["passed"] for e in relevant) / len(relevant)

    def report(self):
        rules = set(e["rule"] for e in self.entries)
        return {rule: f"{self.pass_rate(rule):.1%}" for rule in rules}


class AuditedModule(dspy.Module):
    """Wrapper that logs assertion compliance."""
    def __init__(self, inner_module):
        self.inner = inner_module
        self.log = ComplianceLog()

    def forward(self, **kwargs):
        try:
            result = self.inner(**kwargs)
            # If we got here, all assertions passed
            self.log.log("all_assertions", True)
            return result
        except dspy.primitives.assertions.DSPyAssertionError as e:
            self.log.log("all_assertions", False, str(e))
            raise

# Usage
bot = AuditedModule(BrandCompliantBot())
# ... run many queries ...
print(bot.log.report())
# {"all_assertions": "94.2%"}
```

## Multi-Rule Tweet Writer

Compose five rules on a single output — from the DSPy Assertions paper's TweetGen case study.

```python
class WriteTweet(dspy.Signature):
    """Write an engaging tweet about the topic, incorporating the key facts."""
    topic: str = dspy.InputField()
    key_facts: list[str] = dspy.InputField()
    tweet: str = dspy.OutputField(desc="An engaging tweet, no hashtags, under 280 chars")

class RuleFollowingTweeter(dspy.Module):
    def __init__(self):
        self.write = dspy.ChainOfThought(WriteTweet)

    def forward(self, topic, key_facts):
        result = self.write(topic=topic, key_facts=key_facts)
        tweet = result.tweet

        # Rule 1 (hard): Character limit
        dspy.Assert(len(tweet) <= 280, f"Tweet is {len(tweet)} chars, must be ≤280.")

        # Rule 2 (hard): No hashtags
        dspy.Assert("#" not in tweet, "Remove all hashtags.")

        # Rule 3 (hard): Must include at least one key fact
        dspy.Assert(
            any(fact.lower() in tweet.lower() for fact in key_facts),
            f"Tweet must include at least one of: {key_facts}"
        )

        # Rule 4 (soft): Engaging (not a dry statement)
        dspy.Suggest(
            not tweet.startswith(topic),
            "Don't start with the topic name — make it more engaging."
        )

        # Rule 5 (soft): No URLs (keep it self-contained)
        dspy.Suggest(
            "http" not in tweet,
            "Avoid URLs in the tweet. Make the message self-contained."
        )

        return result

# Usage
tweeter = RuleFollowingTweeter()
result = tweeter(
    topic="Climate Tech",
    key_facts=["Solar costs dropped 90% in 10 years", "Battery storage doubled in capacity"],
)
print(result.tweet)
```
