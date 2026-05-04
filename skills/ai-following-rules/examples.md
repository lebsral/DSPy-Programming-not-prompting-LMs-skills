# Following Rules — Examples

## Content Policy Enforcement

A customer-facing chatbot that must follow brand voice guidelines.

```python
import dspy
import re

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
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

def brand_reward(args: dict, pred: dspy.Prediction) -> float:
    response = pred.response
    score = 1.0

    # Hard rule: word limit (-0.3 per violation)
    word_count = len(response.split())
    if word_count > BRAND_RULES["max_words"]:
        score -= 0.3

    # Hard rule: no blocked phrases (-0.3 per phrase found)
    for phrase in BRAND_RULES["blocked_phrases"]:
        if phrase.lower() in response.lower():
            score -= 0.3

    # Hard rule: no competitor mentions (-0.3)
    if "competitor" in response.lower():
        score -= 0.3

    # Soft rule: include sign-off (-0.1 if missing)
    if not response.strip().endswith(BRAND_RULES["required_sign_off"]):
        score -= 0.1

    return max(score, 0.0)

class BrandCompliantBot(dspy.Module):
    def __init__(self):
        base = dspy.ChainOfThought(BrandResponse)
        self.respond = dspy.Refine(module=base, N=3, reward_fn=brand_reward, threshold=0.8)

    def forward(self, customer_message, context):
        return self.respond(customer_message=customer_message, context=context)

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

def quiz_reward(args: dict, pred: dspy.Prediction) -> float:
    quiz = pred.quiz
    score = 1.0

    # Hard rule: correct answer must be in options (-0.3)
    if quiz.correct_answer not in quiz.options:
        score -= 0.3

    # Hard rule: all options must be unique (-0.3)
    if len(set(quiz.options)) != len(quiz.options):
        score -= 0.3

    # Soft rule: explanation should mention the correct answer (-0.1)
    if quiz.correct_answer.lower() not in quiz.explanation.lower():
        score -= 0.1

    return max(score, 0.0)

class ValidatedQuizGen(dspy.Module):
    def __init__(self):
        base = dspy.ChainOfThought(GenerateQuiz)
        self.generate = dspy.Refine(module=base, N=3, reward_fn=quiz_reward, threshold=0.8)

    def forward(self, topic, difficulty="medium"):
        return self.generate(topic=topic, difficulty=difficulty)

# Usage
gen = ValidatedQuizGen()
result = gen(topic="Python programming", difficulty="medium")
print(result.quiz.model_dump_json(indent=2))
```

## Business Constraint Enforcement

A pricing chatbot that must follow sales rules.

```python
import re

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

def pricing_reward(args: dict, pred: dspy.Prediction) -> float:
    answer = pred.answer
    score = 1.0

    # Hard rule: never invent prices (-0.3 per invented price)
    dollar_amounts = re.findall(r"\$(\d+)", answer)
    valid_prices = {str(v) for v in PRICING.values() if isinstance(v, int)}
    for amount in dollar_amounts:
        if amount not in valid_prices:
            score -= 0.3

    # Hard rule: never offer unauthorized discounts (-0.3 per unauthorized discount)
    discount_mentions = re.findall(r"(\d+)%\s*(?:off|discount)", answer.lower())
    authorized_percents = {str(int(v * 100)) for v in AUTHORIZED_DISCOUNTS.values()}
    for pct in discount_mentions:
        if pct not in authorized_percents:
            score -= 0.3

    # Soft rule: enterprise questions should suggest contacting sales (-0.1)
    question = args.get("question", "")
    if "enterprise" in question.lower():
        if "contact" not in answer.lower() and "sales" not in answer.lower():
            score -= 0.1

    return max(score, 0.0)

class PricingBot(dspy.Module):
    def __init__(self):
        base = dspy.ChainOfThought(PricingAnswer)
        self.respond = dspy.Refine(module=base, N=3, reward_fn=pricing_reward, threshold=0.8)

    def forward(self, question):
        pricing_info = (
            f"Plans: Starter ${PRICING['starter']}/mo, "
            f"Professional ${PRICING['professional']}/mo, "
            f"Enterprise: contact sales. "
            f"Active promotions: {', '.join(AUTHORIZED_DISCOUNTS.keys())}"
        )
        return self.respond(question=question, pricing_info=pricing_info)

# Usage
bot = PricingBot()
result = bot(question="Can I get a discount on the Professional plan?")
print(result.answer)
```

## Compliance Logging

Wrap any rule-following module to log reward scores for auditing.

```python
import time
from dataclasses import dataclass, field

@dataclass
class ComplianceLog:
    """Track reward scores for compliance reporting."""
    entries: list[dict] = field(default_factory=list)

    def log(self, reward_score: float, details: str = ""):
        self.entries.append({
            "timestamp": time.time(),
            "reward_score": reward_score,
            "passed": reward_score >= 0.8,
            "details": details,
        })

    def pass_rate(self) -> float:
        if not self.entries:
            return 0.0
        return sum(e["passed"] for e in self.entries) / len(self.entries)

    def avg_score(self) -> float:
        if not self.entries:
            return 0.0
        return sum(e["reward_score"] for e in self.entries) / len(self.entries)

    def report(self) -> dict:
        return {
            "pass_rate": f"{self.pass_rate():.1%}",
            "avg_reward_score": f"{self.avg_score():.3f}",
            "total_calls": len(self.entries),
        }


class AuditedModule(dspy.Module):
    """Wrapper that logs reward-score compliance for any Refine-based module."""
    def __init__(self, inner_module: dspy.Module, reward_fn, threshold: float = 0.8):
        self.inner = inner_module
        self.reward_fn = reward_fn
        self.threshold = threshold
        self.compliance_log = ComplianceLog()

    def forward(self, **kwargs):
        result = self.inner(**kwargs)
        score = self.reward_fn(kwargs, result)
        self.compliance_log.log(
            reward_score=score,
            details=f"threshold={self.threshold}, passed={score >= self.threshold}",
        )
        return result

# Usage
bot = AuditedModule(
    inner_module=dspy.ChainOfThought(BrandResponse),
    reward_fn=brand_reward,
    threshold=0.8,
)
# ... run many queries ...
print(bot.compliance_log.report())
# {"pass_rate": "94.2%", "avg_reward_score": "0.913", "total_calls": 50}
```

## Multi-Rule Tweet Writer

Enforce five rules on a single output — using `dspy.BestOfN` to pick the highest-scoring attempt.

```python
class WriteTweet(dspy.Signature):
    """Write an engaging tweet about the topic, incorporating the key facts."""
    topic: str = dspy.InputField()
    key_facts: list[str] = dspy.InputField()
    tweet: str = dspy.OutputField(desc="An engaging tweet, no hashtags, under 280 chars")

def tweet_reward(args: dict, pred: dspy.Prediction) -> float:
    tweet = pred.tweet
    key_facts = args.get("key_facts", [])
    topic = args.get("topic", "")
    score = 1.0

    # Rule 1 (hard): character limit (-0.3)
    if len(tweet) > 280:
        score -= 0.3

    # Rule 2 (hard): no hashtags (-0.3)
    if "#" in tweet:
        score -= 0.3

    # Rule 3 (hard): must include at least one key fact (-0.3)
    if not any(fact.lower() in tweet.lower() for fact in key_facts):
        score -= 0.3

    # Rule 4 (soft): don't start with the topic name — make it engaging (-0.1)
    if tweet.startswith(topic):
        score -= 0.1

    # Rule 5 (soft): no URLs — keep it self-contained (-0.1)
    if "http" in tweet:
        score -= 0.1

    return max(score, 0.0)

class RuleFollowingTweeter(dspy.Module):
    def __init__(self):
        base = dspy.ChainOfThought(WriteTweet)
        # BestOfN runs 5 independent attempts and returns the highest-scoring one
        self.write = dspy.BestOfN(module=base, N=5, reward_fn=tweet_reward)

    def forward(self, topic, key_facts):
        return self.write(topic=topic, key_facts=key_facts)

# Usage
tweeter = RuleFollowingTweeter()
result = tweeter(
    topic="Climate Tech",
    key_facts=["Solar costs dropped 90% in 10 years", "Battery storage doubled in capacity"],
)
print(result.tweet)
```
