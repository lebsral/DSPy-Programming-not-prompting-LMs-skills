# Evaluate Examples

## Exact Match Evaluation

A simple QA pipeline evaluated with exact match. Shows the full workflow: setup, devset, metric, evaluation, and inspecting failures.

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Program
qa = dspy.ChainOfThought("question -> answer")

# Devset
devset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is the largest planet in our solar system?", answer="Jupiter").with_inputs("question"),
    dspy.Example(question="Who wrote Romeo and Juliet?", answer="Shakespeare").with_inputs("question"),
    dspy.Example(question="What is the chemical symbol for gold?", answer="Au").with_inputs("question"),
    dspy.Example(question="What year did World War II end?", answer="1945").with_inputs("question"),
    dspy.Example(question="What is the speed of light in m/s?", answer="299792458").with_inputs("question"),
    dspy.Example(question="What is the smallest prime number?", answer="2").with_inputs("question"),
    dspy.Example(question="What language is DSPy written in?", answer="Python").with_inputs("question"),
    dspy.Example(question="How many continents are there?", answer="7").with_inputs("question"),
    dspy.Example(question="What is the boiling point of water in Celsius?", answer="100").with_inputs("question"),
]

# Metric: normalized exact match
def exact_match(example, prediction, trace=None):
    pred = prediction.answer.strip().lower()
    gold = example.answer.strip().lower()
    return pred == gold

# Evaluate
evaluator = Evaluate(
    devset=devset,
    metric=exact_match,
    num_threads=4,
    display_progress=True,
    display_table=5,
    return_all_scores=True,
)

score, all_scores = evaluator(qa)
print(f"\nOverall accuracy: {score:.1f}%")

# Inspect failures
print("\nFailing examples:")
for i, (example, s) in enumerate(zip(devset, all_scores)):
    if not s:
        print(f"  [{i}] Q: {example.question}")
        print(f"       Expected: {example.answer}")
        # Re-run to see what was predicted
        pred = qa(question=example.question)
        print(f"       Got:      {pred.answer}")
```

## LM-as-Judge Evaluation

Grading open-ended answers where exact match is too strict. Uses a separate LM to judge whether predictions are correct and complete.

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Program: explain concepts
explainer = dspy.ChainOfThought("concept -> explanation")

# Devset with reference explanations
devset = [
    dspy.Example(
        concept="photosynthesis",
        explanation="The process by which plants convert sunlight, water, and CO2 into glucose and oxygen.",
    ).with_inputs("concept"),
    dspy.Example(
        concept="recursion in programming",
        explanation="A function that calls itself to solve smaller instances of the same problem, with a base case to stop.",
    ).with_inputs("concept"),
    dspy.Example(
        concept="supply and demand",
        explanation="An economic model where price is determined by the relationship between how much of something is available and how much people want it.",
    ).with_inputs("concept"),
    dspy.Example(
        concept="natural selection",
        explanation="Organisms with traits better suited to their environment are more likely to survive and reproduce, passing those traits on.",
    ).with_inputs("concept"),
    dspy.Example(
        concept="HTTP status codes",
        explanation="Three-digit codes returned by web servers indicating the result of a request: 2xx for success, 4xx for client errors, 5xx for server errors.",
    ).with_inputs("concept"),
]

# LM-as-judge signature
class JudgeExplanation(dspy.Signature):
    """Judge whether the predicted explanation correctly covers the key ideas in the reference explanation. Minor wording differences are fine — focus on factual accuracy and completeness."""
    concept: str = dspy.InputField()
    reference_explanation: str = dspy.InputField(desc="The gold-standard explanation")
    predicted_explanation: str = dspy.InputField(desc="The explanation to evaluate")
    is_correct: bool = dspy.OutputField(desc="True if the prediction covers the key ideas accurately")
    reasoning: str = dspy.OutputField(desc="Brief explanation of the judgment")

# Use a stronger model as the judge
judge_lm = dspy.LM("openai/gpt-4o")

def llm_judge(example, prediction, trace=None):
    judge = dspy.ChainOfThought(JudgeExplanation)
    with dspy.context(lm=judge_lm):
        result = judge(
            concept=example.concept,
            reference_explanation=example.explanation,
            predicted_explanation=prediction.explanation,
        )
    return result.is_correct

# Evaluate
evaluator = Evaluate(
    devset=devset,
    metric=llm_judge,
    num_threads=4,
    display_progress=True,
    display_table=5,
)

score = evaluator(explainer)
print(f"\nJudge accuracy: {score:.1f}%")
```

## Composite Metric Evaluation

Combining correctness, conciseness, and safety into a single weighted score. Demonstrates how to build metrics that balance multiple quality dimensions.

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Program: answer health questions
health_qa = dspy.ChainOfThought("question -> answer")

# Devset
devset = [
    dspy.Example(
        question="What are common symptoms of the flu?",
        answer="Fever, cough, body aches, fatigue, and sometimes vomiting or diarrhea.",
    ).with_inputs("question"),
    dspy.Example(
        question="How much water should an adult drink daily?",
        answer="About 8 cups (2 liters) per day, though needs vary by activity level and climate.",
    ).with_inputs("question"),
    dspy.Example(
        question="What is the recommended amount of sleep for adults?",
        answer="7-9 hours per night.",
    ).with_inputs("question"),
    dspy.Example(
        question="What are the benefits of regular exercise?",
        answer="Improved cardiovascular health, better mood, weight management, stronger bones, and reduced disease risk.",
    ).with_inputs("question"),
    dspy.Example(
        question="What causes seasonal allergies?",
        answer="Immune system overreaction to pollen, mold spores, or other airborne allergens.",
    ).with_inputs("question"),
]

# Safety check signature
class CheckSafety(dspy.Signature):
    """Check if a health answer is safe — does not give specific medical diagnoses, dosage recommendations, or advice to skip professional medical consultation."""
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    is_safe: bool = dspy.OutputField(desc="True if the answer is safe and appropriately general")

# Correctness check signature
class CheckCorrectness(dspy.Signature):
    """Check if the predicted answer captures the key facts from the reference answer."""
    question: str = dspy.InputField()
    reference_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    is_correct: bool = dspy.OutputField(desc="True if key facts are covered accurately")

judge_lm = dspy.LM("openai/gpt-4o")

def composite_metric(example, prediction, trace=None):
    # 1. Correctness (0.6 weight) — LM judge
    with dspy.context(lm=judge_lm):
        correctness_judge = dspy.Predict(CheckCorrectness)
        correctness_result = correctness_judge(
            question=example.question,
            reference_answer=example.answer,
            predicted_answer=prediction.answer,
        )
    correct = float(correctness_result.is_correct)

    # 2. Conciseness (0.2 weight) — heuristic
    word_count = len(prediction.answer.split())
    if word_count <= 50:
        concise = 1.0
    elif word_count <= 100:
        concise = 0.5
    else:
        concise = 0.0

    # 3. Safety (0.2 weight) — LM judge
    with dspy.context(lm=judge_lm):
        safety_judge = dspy.Predict(CheckSafety)
        safety_result = safety_judge(
            question=example.question,
            answer=prediction.answer,
        )
    safe = float(safety_result.is_safe)

    # Weighted composite
    score = 0.6 * correct + 0.2 * concise + 0.2 * safe

    # During optimization, require all three to pass
    if trace is not None:
        return correct and concise >= 0.5 and safe

    return score

# Evaluate
evaluator = Evaluate(
    devset=devset,
    metric=composite_metric,
    num_threads=4,
    display_progress=True,
    display_table=5,
    return_all_scores=True,
)

aggregate, all_scores = evaluator(health_qa)
print(f"\nComposite score: {aggregate:.1f}%")

# Breakdown per example
for i, (example, score) in enumerate(zip(devset, all_scores)):
    status = "PASS" if score >= 0.8 else "WARN" if score >= 0.5 else "FAIL"
    print(f"  [{status}] {example.question[:60]}... score={score:.2f}")
```
