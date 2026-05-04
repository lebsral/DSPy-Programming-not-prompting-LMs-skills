# dspy-rlm Examples

## Example 1: Quality-guided text generation with reward function

Analyze a large corpus of customer reviews to generate a summary, using RLM's code exploration to ensure the summary covers all major themes.

```python
import dspy

# Configure LMs
main_lm = dspy.LM("openai/gpt-4o")
cheap_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=main_lm)

# Suppose we have thousands of customer reviews as a single large string
reviews_text = open("reviews_export.txt").read()  # e.g. 500K chars

# RLM will explore the reviews programmatically rather than
# trying to fit them all in a single prompt
rlm = dspy.RLM(
    "reviews, query -> summary, top_complaints: list[str], sentiment_breakdown: str",
    sub_lm=cheap_lm,
    max_iterations=15,
    verbose=True,
)

result = rlm(
    reviews=reviews_text,
    query="Summarize the key themes, list the top 5 complaints, and give a sentiment breakdown",
)

print("Summary:", result.summary)
print("Top complaints:", result.top_complaints)
print("Sentiment:", result.sentiment_breakdown)

# Inspect how the LM explored the data
for i, step in enumerate(result.trajectory):
    print(f"\n--- Step {i + 1} ---")
    print(f"Code:\n{step['code']}")
    print(f"Output:\n{step['output'][:200]}")
```

What RLM does internally:
1. The LM sees metadata about `reviews` (type: str, length: 500000, preview of first/last chars).
2. It writes code to split reviews by delimiter and count them.
3. It samples batches of reviews and calls `llm_query_batched()` to classify sentiment and extract themes.
4. It aggregates results with Python (counters, sorting).
5. It calls `SUBMIT()` with the structured summary.

### Adding a quality check wrapper

You can wrap RLM in a custom module that validates output quality:

```python
class QualityGuidedAnalysis(dspy.Module):
    def __init__(self):
        self.analyze = dspy.RLM(
            "reviews, query -> summary, top_complaints: list[str]",
            sub_lm=dspy.LM("openai/gpt-4o-mini"),
            max_iterations=15,
        )
        self.judge = dspy.ChainOfThought(
            "query, summary, top_complaints -> quality_score: float, feedback"
        )

    def forward(self, reviews, query):
        result = self.analyze(reviews=reviews, query=query)

        # Score the output
        evaluation = self.judge(
            query=query,
            summary=result.summary,
            top_complaints=result.top_complaints,
        )

        return dspy.Prediction(
            summary=result.summary,
            top_complaints=result.top_complaints,
            quality_score=evaluation.quality_score,
        )

def quality_reward(args, pred):
    """Reward analysis that meets a minimum quality threshold."""
    judge = dspy.ChainOfThought(
        "query, summary, top_complaints -> quality_score: float, feedback"
    )
    evaluation = judge(
        query=args["query"],
        summary=pred.summary,
        top_complaints=pred.top_complaints,
    )
    return float(evaluation.quality_score)

qa = dspy.Refine(
    module=QualityGuidedAnalysis(),
    N=3,
    reward_fn=quality_reward,
    threshold=0.7,
)
result = qa(reviews=reviews_text, query="Summarize customer feedback")
```

---

## Example 2: Constrained generation with reward-based refinement

Analyze server log files to find error patterns, with constraints on output format and completeness.

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o"))

# Custom tool that the LM can call inside the sandbox
def validate_timestamp(ts: str) -> bool:
    """Check if a timestamp string matches ISO 8601 format."""
    from datetime import datetime
    try:
        datetime.fromisoformat(ts)
        return True
    except ValueError:
        return False

# Large log file content
logs = open("/var/log/app/server.log").read()  # e.g. 2M chars

rlm = dspy.RLM(
    "logs, query -> error_patterns: list[str], timeline: str, root_cause: str",
    tools=[validate_timestamp],
    max_iterations=20,
    max_llm_calls=30,
    verbose=True,
)

result = rlm(
    logs=logs,
    query="Identify recurring error patterns, build a timeline of incidents, and suggest root causes",
)

print("Error patterns:", result.error_patterns)
print("Timeline:", result.timeline)
print("Root cause:", result.root_cause)
```

### Wrapping with constraints for production use

Combine RLM with a reward function to enforce output requirements:

```python
class ConstrainedLogAnalysis(dspy.Module):
    def __init__(self):
        self.analyze = dspy.RLM(
            "logs, query -> error_patterns: list[str], timeline: str, root_cause: str",
            max_iterations=20,
            max_llm_calls=30,
        )

    def forward(self, logs, query):
        return self.analyze(logs=logs, query=query)


def log_analysis_reward(args, pred):
    """Hard constraints return 0.0 on failure; soft constraints subtract a small penalty."""
    # Hard constraints - must have patterns and substantive root cause
    if len(pred.error_patterns) == 0:
        return 0.0
    if len(pred.root_cause) < 20:
        return 0.0
    # Soft constraint - prefer focused pattern lists
    score = 1.0
    if len(pred.error_patterns) > 10:
        score -= 0.1
    return score


analyzer = dspy.Refine(
    module=ConstrainedLogAnalysis(),
    N=3,
    reward_fn=log_analysis_reward,
    threshold=0.9,
)


# Use with evaluation
def completeness_metric(example, pred, trace=None):
    """Score based on whether all required fields are populated and useful."""
    has_patterns = len(pred.error_patterns) > 0
    has_timeline = len(pred.timeline) > 50
    has_root_cause = len(pred.root_cause) > 20
    return (has_patterns + has_timeline + has_root_cause) / 3.0


result = analyzer(
    logs=logs,
    query="Find error patterns and root causes from the last 24 hours",
)
```

### Using RLM with an optimizer

RLM modules can be optimized like any other DSPy module:

```python
from dspy.evaluate import Evaluate

# Prepare labeled examples
trainset = [
    dspy.Example(
        logs=open(f"logs/sample_{i}.txt").read(),
        query="Identify error patterns and root causes",
        error_patterns=expected_patterns[i],
        root_cause=expected_causes[i],
    ).with_inputs("logs", "query")
    for i in range(50)
]

# Evaluate baseline
evaluator = Evaluate(devset=trainset[:10], metric=completeness_metric, num_threads=2)
baseline_score = evaluator(analyzer)
print(f"Baseline: {baseline_score}")

# Optimize
optimizer = dspy.BootstrapFewShot(metric=completeness_metric, max_bootstrapped_demos=3)
optimized = optimizer.compile(analyzer, trainset=trainset)

optimized_score = evaluator(optimized)
print(f"Optimized: {optimized_score}")

optimized.save("optimized_log_analyzer.json")
```
