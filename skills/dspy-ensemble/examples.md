# dspy-ensemble -- Worked Examples

## Example 1: Majority voting ensemble from multiple optimization runs

Run BootstrapFewShot three times with different random seeds, then combine the optimized programs with majority voting. This is the most common Ensemble pattern -- it smooths out the randomness in optimization and gives more reliable answers.

```python
import dspy
from dspy.evaluate import Evaluate


# --- Signature ---

class FactualQA(dspy.Signature):
    """Answer the question with a short factual response."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="A short factual answer")


# --- Metric ---

def exact_match(example, pred, trace=None):
    return pred.answer.strip().lower() == example.answer.strip().lower()


# --- Dataset ---

trainset = [
    dspy.Example(question="What is the chemical symbol for gold?", answer="Au"),
    dspy.Example(question="How many continents are there?", answer="7"),
    dspy.Example(question="What planet is closest to the Sun?", answer="Mercury"),
    dspy.Example(question="What is the boiling point of water in Celsius?", answer="100"),
    dspy.Example(question="Who wrote Romeo and Juliet?", answer="Shakespeare"),
    dspy.Example(question="What is the square root of 144?", answer="12"),
    dspy.Example(question="What gas do plants absorb from the atmosphere?", answer="Carbon dioxide"),
    dspy.Example(question="What is the largest ocean on Earth?", answer="Pacific"),
    dspy.Example(question="How many sides does a hexagon have?", answer="6"),
    dspy.Example(question="What is the freezing point of water in Fahrenheit?", answer="32"),
]
trainset = [ex.with_inputs("question") for ex in trainset]

devset = [
    dspy.Example(question="What is the capital of Japan?", answer="Tokyo"),
    dspy.Example(question="How many legs does a spider have?", answer="8"),
    dspy.Example(question="What element does O represent?", answer="Oxygen"),
    dspy.Example(question="What is the smallest prime number?", answer="2"),
    dspy.Example(question="What continent is Brazil on?", answer="South America"),
]
devset = [ex.with_inputs("question") for ex in devset]


# --- Setup ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# --- Step 1: Run multiple optimization passes ---

base_program = dspy.ChainOfThought(FactualQA)

optimized_programs = []
for run_idx in range(3):
    optimizer = dspy.BootstrapFewShot(
        metric=exact_match,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
    )
    optimized = optimizer.compile(base_program, trainset=trainset)
    optimized_programs.append(optimized)
    print(f"Run {run_idx + 1} compiled.")


# --- Step 2: Combine with Ensemble ---

ensemble_optimizer = dspy.Ensemble(reduce_fn=dspy.majority)
ensemble_program = ensemble_optimizer.compile(optimized_programs)


# --- Step 3: Evaluate the ensemble vs individual programs ---

evaluator = Evaluate(
    devset=devset,
    metric=exact_match,
    num_threads=4,
    display_progress=True,
)

# Score each individual program
for i, prog in enumerate(optimized_programs):
    score = evaluator(prog)
    print(f"Program {i + 1} accuracy: {score:.1f}%")

# Score the ensemble
ensemble_score = evaluator(ensemble_program)
print(f"Ensemble accuracy: {ensemble_score:.1f}%")


# --- Step 4: Use the ensemble in production ---

result = ensemble_program(question="What is the capital of Germany?")
print(f"Answer: {result.answer}")
```

Key points:
- Each BootstrapFewShot run picks different demonstrations due to randomness, so the three programs have different strengths
- `dspy.majority` counts votes across all three programs and returns the most common answer
- The ensemble typically matches or beats the best individual program because voting corrects occasional errors from any single program
- Evaluate the ensemble on a devset to confirm the improvement is real before deploying


## Example 2: Ensemble with different model configurations

Combine programs optimized with different LMs or optimization strategies. Each model brings different capabilities -- a cheaper model may be fast and often correct, while a more capable model catches harder cases. Ensembling them via majority voting gives you the best of both.

```python
import dspy
from dspy.evaluate import Evaluate


# --- Signature ---

class ClassifyIntent(dspy.Signature):
    """Classify the user message into one of the given intent categories."""
    message: str = dspy.InputField(desc="User message to classify")
    categories: str = dspy.InputField(desc="Comma-separated list of valid categories")
    intent: str = dspy.OutputField(desc="The best matching category")


# --- Metric ---

def correct_intent(example, pred, trace=None):
    return pred.intent.strip().lower() == example.intent.strip().lower()


# --- Dataset ---

categories = "billing, technical_support, account, general_inquiry, cancellation"

trainset = [
    dspy.Example(message="I was charged twice this month", categories=categories, intent="billing"),
    dspy.Example(message="My app keeps crashing on startup", categories=categories, intent="technical_support"),
    dspy.Example(message="How do I change my password?", categories=categories, intent="account"),
    dspy.Example(message="What are your business hours?", categories=categories, intent="general_inquiry"),
    dspy.Example(message="I want to cancel my subscription", categories=categories, intent="cancellation"),
    dspy.Example(message="The invoice amount is wrong", categories=categories, intent="billing"),
    dspy.Example(message="I can't connect to the API", categories=categories, intent="technical_support"),
    dspy.Example(message="Update my email address please", categories=categories, intent="account"),
    dspy.Example(message="Do you offer student discounts?", categories=categories, intent="general_inquiry"),
    dspy.Example(message="Please stop my auto-renewal", categories=categories, intent="cancellation"),
]
trainset = [ex.with_inputs("message", "categories") for ex in trainset]

devset = [
    dspy.Example(message="Why is my bill higher this month?", categories=categories, intent="billing"),
    dspy.Example(message="The website shows a 500 error", categories=categories, intent="technical_support"),
    dspy.Example(message="I need to update my shipping address", categories=categories, intent="account"),
    dspy.Example(message="What payment methods do you accept?", categories=categories, intent="general_inquiry"),
    dspy.Example(message="I'd like to close my account", categories=categories, intent="cancellation"),
]
devset = [ex.with_inputs("message", "categories") for ex in devset]


# --- Setup: two different LMs ---

fast_lm = dspy.LM("openai/gpt-4o-mini")
strong_lm = dspy.LM("openai/gpt-4o")


# --- Step 1: Optimize a program with the fast LM ---

dspy.configure(lm=fast_lm)
base_program = dspy.ChainOfThought(ClassifyIntent)

opt_fast = dspy.BootstrapFewShot(
    metric=correct_intent,
    max_bootstrapped_demos=4,
)
prog_fast = opt_fast.compile(base_program, trainset=trainset)
print("Fast-model program compiled.")


# --- Step 2: Optimize a program with the strong LM ---

dspy.configure(lm=strong_lm)
base_program_strong = dspy.ChainOfThought(ClassifyIntent)

opt_strong = dspy.MIPROv2(
    metric=correct_intent,
    auto="light",
)
prog_strong = opt_strong.compile(base_program_strong, trainset=trainset)
print("Strong-model program compiled.")


# --- Step 3: Optimize another variant with random search ---

dspy.configure(lm=fast_lm)
base_program_rs = dspy.ChainOfThought(ClassifyIntent)

opt_rs = dspy.BootstrapFewShotWithRandomSearch(
    metric=correct_intent,
    max_bootstrapped_demos=4,
    num_candidate_programs=5,
)
prog_rs = opt_rs.compile(base_program_rs, trainset=trainset)
print("Random-search program compiled.")


# --- Step 4: Ensemble all three ---

ensemble_optimizer = dspy.Ensemble(reduce_fn=dspy.majority, size=None)
ensemble_program = ensemble_optimizer.compile([prog_fast, prog_strong, prog_rs])


# --- Step 5: Evaluate ---

dspy.configure(lm=fast_lm)  # LM for evaluation context

evaluator = Evaluate(
    devset=devset,
    metric=correct_intent,
    num_threads=4,
    display_progress=True,
)

score_fast = evaluator(prog_fast)
print(f"Fast-model program accuracy: {score_fast:.1f}%")

score_strong = evaluator(prog_strong)
print(f"Strong-model program accuracy: {score_strong:.1f}%")

score_rs = evaluator(prog_rs)
print(f"Random-search program accuracy: {score_rs:.1f}%")

ensemble_score = evaluator(ensemble_program)
print(f"Ensemble accuracy: {ensemble_score:.1f}%")


# --- Step 6: Use in production ---

result = ensemble_program(
    message="I see an unauthorized charge on my credit card",
    categories=categories,
)
print(f"Intent: {result.intent}")
```

Key points:
- Each program uses a different optimization strategy and potentially a different LM, creating genuine diversity in how they approach the task
- The fast-model program (gpt-4o-mini + BootstrapFewShot) is cheap and handles easy cases well
- The strong-model program (gpt-4o + MIPROv2) handles edge cases and ambiguous inputs better
- The random-search program explores a wider space of few-shot demonstrations
- Majority voting across all three is more reliable than any single program because errors from different strategies are unlikely to be correlated
- In production, you pay for all three LM calls per input -- use `size=2` if you want to reduce cost by sampling a subset each time
