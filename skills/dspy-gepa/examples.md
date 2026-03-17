# dspy.GEPA Examples

## Example 1: Instruction optimization for classification

A sentiment classifier optimized with GEPA. The feedback metric tells the reflection LM exactly what went wrong when the classifier mislabels an example, so GEPA can propose instructions that address common failure patterns like sarcasm or mixed sentiment.

```python
from typing import Literal
import dspy
from dspy.evaluate import Evaluate

# Configure LMs
task_lm = dspy.LM("openai/gpt-4o-mini")
reflection_lm = dspy.LM("openai/gpt-4o", temperature=1.0, max_tokens=4096)
dspy.configure(lm=task_lm)

# Define the classifier
class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of a customer review."""
    review: str = dspy.InputField(desc="A customer review")
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField(
        desc="The sentiment of the review"
    )

classify = dspy.ChainOfThought(SentimentClassifier)

# Prepare training data (~50 examples)
trainset = [
    dspy.Example(
        review="Absolutely love this product! Best purchase I've made all year.",
        sentiment="positive",
    ).with_inputs("review"),
    dspy.Example(
        review="Broke after two days. Complete waste of money.",
        sentiment="negative",
    ).with_inputs("review"),
    dspy.Example(
        review="It works fine. Nothing special but gets the job done.",
        sentiment="neutral",
    ).with_inputs("review"),
    dspy.Example(
        review="Oh sure, because crashing every five minutes is a 'feature'.",
        sentiment="negative",
    ).with_inputs("review"),
    dspy.Example(
        review="The packaging was nice but the product itself is mediocre.",
        sentiment="neutral",
    ).with_inputs("review"),
    # ... add more examples to reach ~50
]

# Hold out some examples for validation
valset = trainset[40:]
trainset = trainset[:40]


# Define a feedback metric
def sentiment_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Score the prediction and provide actionable feedback on failures."""
    correct = pred.sentiment == gold.sentiment

    if correct:
        return {"score": 1.0, "feedback": ""}

    # Build targeted feedback for the reflection LM
    feedback_parts = []

    # Detect common failure patterns
    review_lower = gold.review.lower()
    if gold.sentiment == "negative" and pred.sentiment == "positive":
        if any(word in review_lower for word in ["sure", "great", "love", "amazing"]):
            feedback_parts.append(
                "This review uses sarcasm -- positive words with negative intent. "
                "The instruction should tell the model to watch for sarcastic tone."
            )
        else:
            feedback_parts.append(
                f"Misclassified negative review as positive. "
                f"The review expresses dissatisfaction."
            )

    if gold.sentiment == "neutral" and pred.sentiment != "neutral":
        feedback_parts.append(
            "This review has mixed or mild signals. The instruction should clarify "
            "that reviews without strong positive or negative language are neutral."
        )

    if gold.sentiment == "positive" and pred.sentiment == "negative":
        feedback_parts.append(
            "Misclassified a genuinely positive review as negative. "
            "The instruction should not over-correct for sarcasm."
        )

    if not feedback_parts:
        feedback_parts.append(
            f"Expected '{gold.sentiment}' but predicted '{pred.sentiment}'. "
            f"Review: '{gold.review[:80]}...'"
        )

    return {"score": 0.0, "feedback": " ".join(feedback_parts)}


# Evaluate baseline
evaluator = Evaluate(devset=valset, metric=sentiment_metric, num_threads=4)
baseline_score = evaluator(classify)
print(f"Baseline score: {baseline_score}")

# Optimize with GEPA
gepa = dspy.GEPA(
    metric=sentiment_metric,
    reflection_lm=reflection_lm,
    auto="medium",
)
optimized_classify = gepa.compile(classify, trainset=trainset, valset=valset)

# Evaluate optimized program
optimized_score = evaluator(optimized_classify)
print(f"Optimized score: {optimized_score}")
print(f"Improvement: {baseline_score} -> {optimized_score}")

# Save the optimized program
optimized_classify.save("optimized_sentiment.json")

# Use it
result = optimized_classify(review="Yeah right, 'premium quality' that falls apart in a week.")
print(f"Sentiment: {result.sentiment}")
print(f"Reasoning: {result.reasoning}")
```

What this demonstrates:

- **Feedback metric with failure analysis** -- the metric detects sarcasm, mixed signals, and over-correction patterns, giving the reflection LM concrete guidance
- **Structured feedback** -- instead of just "wrong", the feedback says _why_ the instruction should change (e.g., "watch for sarcastic tone")
- **Baseline comparison** -- evaluating before and after GEPA to measure the actual improvement
- **Class-based signature** -- `SentimentClassifier` with typed `Literal` output constrains the label space
- **Separate validation set** -- prevents overfitting by holding out examples from training

## Example 2: Instruction tuning for a generation task

A summarization pipeline where GEPA optimizes instructions based on multiple quality dimensions. The feedback metric scores summaries on faithfulness, conciseness, and completeness, giving per-dimension feedback so the reflection LM knows which aspect of the instruction to improve.

```python
import dspy
from dspy.evaluate import Evaluate

# Configure LMs
task_lm = dspy.LM("openai/gpt-4o-mini")
reflection_lm = dspy.LM("openai/gpt-4o", temperature=1.0, max_tokens=4096)
dspy.configure(lm=task_lm)


# Define a two-step summarization pipeline
class ExtractKeyPoints(dspy.Signature):
    """Extract the key points from an article."""
    article: str = dspy.InputField(desc="The full article text")
    key_points: str = dspy.OutputField(desc="Bullet-pointed list of key facts")


class WriteSummary(dspy.Signature):
    """Write a concise summary from key points."""
    key_points: str = dspy.InputField(desc="Extracted key points")
    summary: str = dspy.OutputField(desc="A 2-3 sentence summary")


class Summarizer(dspy.Module):
    def __init__(self):
        self.extract = dspy.ChainOfThought(ExtractKeyPoints)
        self.summarize = dspy.ChainOfThought(WriteSummary)

    def forward(self, article):
        extraction = self.extract(article=article)
        return self.summarize(key_points=extraction.key_points)


# Prepare training data
# Each example has an article and a reference summary
trainset = [
    dspy.Example(
        article=(
            "Researchers at MIT have developed a new battery technology that "
            "could double the range of electric vehicles. The solid-state "
            "battery uses a lithium-metal anode and a ceramic electrolyte, "
            "eliminating the risk of fire associated with liquid electrolytes. "
            "The team published their findings in Nature Energy and expects "
            "commercial production within five years."
        ),
        reference_summary=(
            "MIT researchers created a solid-state battery with a lithium-metal "
            "anode and ceramic electrolyte that could double EV range while "
            "eliminating fire risk. Commercial production is expected within "
            "five years."
        ),
    ).with_inputs("article"),
    dspy.Example(
        article=(
            "The European Central Bank raised interest rates by 0.25 percentage "
            "points to 4.5%, marking the tenth consecutive increase. ECB "
            "President Christine Lagarde cited persistent inflation in services "
            "and food prices. Markets reacted negatively, with the Euro Stoxx "
            "50 falling 1.2% on the announcement."
        ),
        reference_summary=(
            "The ECB raised rates by 0.25 points to 4.5% for the tenth "
            "straight increase, citing persistent inflation. European stocks "
            "fell 1.2% in response."
        ),
    ).with_inputs("article"),
    # ... add more examples to reach ~50
]

valset = trainset[40:]
trainset = trainset[:40]


# Define a multi-dimensional feedback metric
def summary_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Score summaries on faithfulness, conciseness, and completeness."""
    summary = pred.summary
    article = gold.article
    reference = gold.reference_summary

    score = 0.0
    feedback_parts = []

    # Dimension 1: Conciseness (0.3 points)
    # Summary should be shorter than 40% of the article
    summary_words = len(summary.split())
    article_words = len(article.split())
    ratio = summary_words / max(article_words, 1)

    if ratio <= 0.4:
        score += 0.3
    elif ratio <= 0.6:
        score += 0.15
        feedback_parts.append(
            f"Summary is {summary_words} words ({ratio:.0%} of article). "
            f"Aim for under 40% of the article length. "
            f"The instruction should emphasize brevity."
        )
    else:
        feedback_parts.append(
            f"Summary is too long ({summary_words} words, {ratio:.0%} of article). "
            f"The instruction should strongly emphasize conciseness and "
            f"mention a target of 2-3 sentences."
        )

    # Dimension 2: Completeness (0.4 points)
    # Check if key terms from the reference appear in the summary
    ref_words = set(reference.lower().split())
    summary_words_set = set(summary.lower().split())
    # Filter to meaningful words (>4 chars)
    key_ref_words = {w for w in ref_words if len(w) > 4}
    if key_ref_words:
        overlap = len(key_ref_words & summary_words_set) / len(key_ref_words)
        score += 0.4 * overlap
        if overlap < 0.5:
            missing = key_ref_words - summary_words_set
            feedback_parts.append(
                f"Summary misses key information. Missing terms: "
                f"{', '.join(list(missing)[:5])}. "
                f"The instruction should emphasize capturing all main facts."
            )

    # Dimension 3: Faithfulness (0.3 points)
    # Basic check: summary should not introduce words absent from article
    article_words_set = set(article.lower().split())
    summary_unique = set(summary.lower().split())
    novel_words = summary_unique - article_words_set
    # Filter to meaningful novel words
    novel_meaningful = {w for w in novel_words if len(w) > 5 and w.isalpha()}
    if len(novel_meaningful) <= 2:
        score += 0.3
    elif len(novel_meaningful) <= 5:
        score += 0.15
        feedback_parts.append(
            f"Summary introduces terms not in the article: "
            f"{', '.join(list(novel_meaningful)[:3])}. "
            f"The instruction should say to only use information from the source."
        )
    else:
        feedback_parts.append(
            f"Summary adds too much information not in the article. "
            f"Novel terms: {', '.join(list(novel_meaningful)[:5])}. "
            f"The instruction must emphasize faithfulness to the source text."
        )

    # Per-predictor feedback for multi-step pipeline
    if pred_name == "extract" and feedback_parts:
        feedback_parts.insert(
            0, "The key-point extraction step may be missing important facts. "
        )
    elif pred_name == "summarize" and feedback_parts:
        feedback_parts.insert(
            0, "The summary-writing step needs improvement. "
        )

    feedback = " ".join(feedback_parts) if feedback_parts else ""
    return {"score": score, "feedback": feedback}


# Evaluate baseline
summarizer = Summarizer()
evaluator = Evaluate(devset=valset, metric=summary_metric, num_threads=4)
baseline_score = evaluator(summarizer)
print(f"Baseline score: {baseline_score}")

# Optimize with GEPA
gepa = dspy.GEPA(
    metric=summary_metric,
    reflection_lm=reflection_lm,
    auto="medium",
    reflection_minibatch_size=5,  # more context per reflection for generation
)
optimized_summarizer = gepa.compile(summarizer, trainset=trainset, valset=valset)

# Evaluate optimized pipeline
optimized_score = evaluator(optimized_summarizer)
print(f"Optimized score: {optimized_score}")
print(f"Improvement: {baseline_score} -> {optimized_score}")

# Save the optimized pipeline
optimized_summarizer.save("optimized_summarizer.json")

# Use it
result = optimized_summarizer(
    article=(
        "SpaceX successfully launched its Starship rocket on its third test "
        "flight, reaching orbital velocity for the first time. The vehicle "
        "re-entered the atmosphere but broke apart before landing. CEO Elon "
        "Musk called it a 'huge step forward' and said the next flight would "
        "attempt a controlled ocean landing within three months."
    )
)
print(f"Summary: {result.summary}")
```

What this demonstrates:

- **Multi-step pipeline** -- GEPA optimizes instructions for both `extract` and `summarize` predictors independently via round-robin component selection
- **Per-predictor feedback** -- the metric uses `pred_name` to tailor feedback to the specific step being optimized
- **Multi-dimensional scoring** -- conciseness (0.3), completeness (0.4), and faithfulness (0.3) are scored separately with targeted feedback for each
- **Actionable feedback per dimension** -- instead of just "bad summary", the metric says exactly which dimension failed and what the instruction should emphasize
- **Larger minibatch** -- `reflection_minibatch_size=5` gives the reflection LM more context for generation tasks where quality is subjective
- **Baseline comparison** -- measuring improvement before and after optimization to validate the effort
