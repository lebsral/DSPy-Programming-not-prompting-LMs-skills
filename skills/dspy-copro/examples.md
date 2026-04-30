# dspy.COPRO Examples

## Example 1: Instruction optimization with breadth search

A sentiment classification pipeline where COPRO searches across many instruction candidates to find the phrasing that maximizes accuracy. Demonstrates tuning `breadth` and inspecting candidate results.

```python
import dspy
from dspy.evaluate import Evaluate

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

# Define a classification signature with an initial instruction
class ClassifySentiment(dspy.Signature):
    """Classify the sentiment of the given text."""
    text: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="One of: positive, negative, neutral")


# Build the program
classify = dspy.ChainOfThought(ClassifySentiment)

# Prepare training and dev data
trainset = [
    dspy.Example(text="This product changed my life for the better!", sentiment="positive").with_inputs("text"),
    dspy.Example(text="Worst purchase I have ever made.", sentiment="negative").with_inputs("text"),
    dspy.Example(text="It works as described, nothing special.", sentiment="neutral").with_inputs("text"),
    dspy.Example(text="Absolutely love the build quality.", sentiment="positive").with_inputs("text"),
    dspy.Example(text="Broke after two days of normal use.", sentiment="negative").with_inputs("text"),
    dspy.Example(text="Average product for the price point.", sentiment="neutral").with_inputs("text"),
    dspy.Example(text="Five stars, exceeded all expectations!", sentiment="positive").with_inputs("text"),
    dspy.Example(text="Customer support was unhelpful and rude.", sentiment="negative").with_inputs("text"),
    dspy.Example(text="Shipping was on time, product is okay.", sentiment="neutral").with_inputs("text"),
    dspy.Example(text="I recommend this to everyone I know.", sentiment="positive").with_inputs("text"),
    # ... add more examples for better results (50-200 recommended)
]

devset = [
    dspy.Example(text="The fabric feels cheap but the design is nice.", sentiment="neutral").with_inputs("text"),
    dspy.Example(text="Never buying from this brand again.", sentiment="negative").with_inputs("text"),
    dspy.Example(text="My kids love it, great gift idea!", sentiment="positive").with_inputs("text"),
]


# Define the metric
def sentiment_match(example, prediction, trace=None):
    return prediction.sentiment.strip().lower() == example.sentiment.strip().lower()


# Evaluate the baseline (before optimization)
evaluator = Evaluate(devset=devset, metric=sentiment_match, num_threads=4)
baseline_score = evaluator(classify)
print(f"Baseline score: {baseline_score}")

# Optimize with COPRO -- wide breadth to explore many instruction variants
optimizer = dspy.COPRO(
    metric=sentiment_match,
    breadth=20,            # Generate 19 candidates + 1 base per round
    depth=3,               # 3 rounds of refinement
    init_temperature=1.4,  # Diverse candidate generation
    track_stats=True,      # Log per-iteration statistics
)

optimized = optimizer.compile(
    classify,
    trainset=trainset,
    eval_kwargs=dict(num_threads=4, display_progress=True),
)

# Evaluate the optimized program
optimized_score = evaluator(optimized)
print(f"Optimized score: {optimized_score}")
print(f"Improvement: {optimized_score - baseline_score}")

# Inspect what instructions COPRO tried
for predictor_name, candidates in optimized.candidate_programs.items():
    print(f"\n--- Candidates for {predictor_name} ---")
    # Sort by score to see top performers
    sorted_candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    for i, candidate in enumerate(sorted_candidates[:5]):
        print(f"\n  #{i+1} (score: {candidate['score']:.2f})")
        print(f"  Instruction: {candidate['instruction'][:120]}...")

# Use the optimized program
result = optimized(text="The battery life is phenomenal, best I have seen in years.")
print(f"\nPrediction: {result.sentiment}")
print(f"Reasoning: {result.reasoning}")

# Save for production
optimized.save("optimized_classifier.json")
```

What this demonstrates:

- **Wide breadth search (20)** -- generates 19 alternative instructions per round, increasing the chance of finding a high-performing phrasing
- **Tracking statistics** -- `track_stats=True` logs per-iteration performance to monitor convergence
- **Inspecting candidates** -- after optimization, `candidate_programs` contains every instruction tried and its score, letting you understand what worked
- **Baseline comparison** -- always evaluate before and after to confirm the optimization actually helped
- **Class-based signature** -- the initial instruction ("Classify the sentiment of the given text.") is the docstring, which COPRO replaces with better alternatives

## Example 2: Comparing COPRO candidates across configurations

Run COPRO with different breadth settings to understand the cost-quality tradeoff. This pattern helps you decide on the right breadth for your task before committing to a full optimization run.

```python
import dspy
from dspy.evaluate import Evaluate

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

# A question-answering program
class AnswerQuestion(dspy.Signature):
    """Answer the question based on general knowledge."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="A concise factual answer")


# Training and dev data
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="Who wrote Romeo and Juliet?", answer="Shakespeare").with_inputs("question"),
    dspy.Example(question="What planet is closest to the Sun?", answer="Mercury").with_inputs("question"),
    dspy.Example(question="What is the chemical symbol for gold?", answer="Au").with_inputs("question"),
    dspy.Example(question="In what year did World War II end?", answer="1945").with_inputs("question"),
    # ... add more for reliable results
]

devset = [
    dspy.Example(question="What is the largest ocean?", answer="Pacific").with_inputs("question"),
    dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
    dspy.Example(question="What is the boiling point of water in Celsius?", answer="100").with_inputs("question"),
]


def answer_match(example, prediction, trace=None):
    return example.answer.lower() in prediction.answer.lower()


evaluator = Evaluate(devset=devset, metric=answer_match, num_threads=4)

# Compare different breadth settings
configs = [
    {"breadth": 5,  "depth": 2, "label": "narrow (breadth=5, depth=2)"},
    {"breadth": 10, "depth": 3, "label": "default (breadth=10, depth=3)"},
    {"breadth": 25, "depth": 3, "label": "wide (breadth=25, depth=3)"},
]

results = []

for config in configs:
    print(f"\n{'='*60}")
    print(f"Running: {config['label']}")
    print(f"{'='*60}")

    # Fresh program for each run
    program = dspy.ChainOfThought(AnswerQuestion)

    optimizer = dspy.COPRO(
        metric=answer_match,
        breadth=config["breadth"],
        depth=config["depth"],
        track_stats=True,
    )

    optimized = optimizer.compile(
        program,
        trainset=trainset,
        eval_kwargs=dict(num_threads=4, display_progress=True),
    )

    score = evaluator(optimized)
    total_calls = optimized.total_calls

    results.append({
        "label": config["label"],
        "score": score,
        "total_calls": total_calls,
        "optimized_program": optimized,
    })

    print(f"Score: {score:.1f}% | LM calls: {total_calls}")

# Print comparison table
print(f"\n{'='*60}")
print(f"{'Config':<40} {'Score':>8} {'LM Calls':>10}")
print(f"{'-'*60}")
for r in results:
    print(f"{r['label']:<40} {r['score']:>7.1f}% {r['total_calls']:>10}")

# Show the winning instruction from the best config
best = max(results, key=lambda r: r["score"])
print(f"\nBest config: {best['label']} ({best['score']:.1f}%)")

# Inspect the winning instruction
for predictor_name, candidates in best["optimized_program"].candidate_programs.items():
    top = max(candidates, key=lambda c: c["score"])
    print(f"\nBest instruction for {predictor_name}:")
    print(f"  \"{top['instruction']}\"")
    print(f"  Score: {top['score']:.2f}")
```

What this demonstrates:

- **Breadth comparison** -- runs the same task with breadth=5, 10, and 25 to show the tradeoff between search coverage and cost
- **Cost tracking** -- `total_calls` shows how many LM calls each configuration used, making the cost difference concrete
- **Fresh program per run** -- creates a new `dspy.ChainOfThought` for each configuration to ensure a fair comparison
- **Extracting the winning instruction** -- after comparing configs, inspects the best-scoring instruction to see what COPRO found
- **Practical decision-making** -- this pattern helps you choose the right breadth setting for your task before committing to a production optimization run
