# BootstrapFewShot Examples

## QA Optimization with Exact Match Metric

A question-answering pipeline optimized with BootstrapFewShot. Shows the full workflow: baseline evaluation, optimization, and before/after comparison.

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Program
qa = dspy.ChainOfThought("question -> answer")

# Training set (~50 examples for bootstrapping)
trainset = [
    dspy.Example(question="What is the chemical symbol for water?", answer="H2O").with_inputs("question"),
    dspy.Example(question="What planet is closest to the Sun?", answer="Mercury").with_inputs("question"),
    dspy.Example(question="What is the square root of 144?", answer="12").with_inputs("question"),
    dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
    dspy.Example(question="What is the capital of Japan?", answer="Tokyo").with_inputs("question"),
    dspy.Example(question="How many legs does a spider have?", answer="8").with_inputs("question"),
    dspy.Example(question="What gas do plants absorb from the atmosphere?", answer="Carbon dioxide").with_inputs("question"),
    dspy.Example(question="What is the largest ocean on Earth?", answer="Pacific Ocean").with_inputs("question"),
    dspy.Example(question="Who developed the theory of relativity?", answer="Albert Einstein").with_inputs("question"),
    dspy.Example(question="What is the freezing point of water in Fahrenheit?", answer="32").with_inputs("question"),
    dspy.Example(question="What is the powerhouse of the cell?", answer="Mitochondria").with_inputs("question"),
    dspy.Example(question="How many bones are in the adult human body?", answer="206").with_inputs("question"),
    dspy.Example(question="What is the hardest natural substance?", answer="Diamond").with_inputs("question"),
    dspy.Example(question="What language has the most native speakers?", answer="Mandarin Chinese").with_inputs("question"),
    dspy.Example(question="What is the speed of sound in m/s at sea level?", answer="343").with_inputs("question"),
    dspy.Example(question="Who wrote The Great Gatsby?", answer="F. Scott Fitzgerald").with_inputs("question"),
    dspy.Example(question="What is the atomic number of carbon?", answer="6").with_inputs("question"),
    dspy.Example(question="What is the tallest mountain on Earth?", answer="Mount Everest").with_inputs("question"),
    dspy.Example(question="What year was the internet invented?", answer="1969").with_inputs("question"),
    dspy.Example(question="What is the currency of the United Kingdom?", answer="Pound sterling").with_inputs("question"),
]

# Held-out dev set (never used for training)
devset = [
    dspy.Example(question="What is the capital of Australia?", answer="Canberra").with_inputs("question"),
    dspy.Example(question="What element does 'O' represent on the periodic table?", answer="Oxygen").with_inputs("question"),
    dspy.Example(question="How many sides does a hexagon have?", answer="6").with_inputs("question"),
    dspy.Example(question="Who invented the telephone?", answer="Alexander Graham Bell").with_inputs("question"),
    dspy.Example(question="What is the largest mammal?", answer="Blue whale").with_inputs("question"),
    dspy.Example(question="What is the boiling point of water in Celsius?", answer="100").with_inputs("question"),
    dspy.Example(question="What continent is Brazil on?", answer="South America").with_inputs("question"),
    dspy.Example(question="How many planets are in the solar system?", answer="8").with_inputs("question"),
    dspy.Example(question="Who wrote 1984?", answer="George Orwell").with_inputs("question"),
    dspy.Example(question="What is the chemical symbol for gold?", answer="Au").with_inputs("question"),
]

# Metric: normalized exact match
def exact_match(example, prediction, trace=None):
    pred = prediction.answer.strip().lower()
    gold = example.answer.strip().lower()
    return pred == gold

# Evaluate baseline
evaluator = Evaluate(
    devset=devset,
    metric=exact_match,
    num_threads=4,
    display_progress=True,
    display_table=5,
)
baseline_score = evaluator(qa)
print(f"Baseline: {baseline_score:.1f}%")

# Optimize with BootstrapFewShot
optimizer = dspy.BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)
optimized_qa = optimizer.compile(qa, trainset=trainset)

# Evaluate optimized program
optimized_score = evaluator(optimized_qa)
print(f"Optimized: {optimized_score:.1f}%")
print(f"Delta: {optimized_score - baseline_score:+.1f}%")

# Save for production use
optimized_qa.save("optimized_qa.json")
```

## Classification with Bootstrapped Demos

A sentiment classifier optimized with BootstrapFewShot. Demonstrates typed outputs, a classification metric, and trace-aware filtering.

```python
import dspy
from typing import Literal
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Signature with typed output
class ClassifySentiment(dspy.Signature):
    """Classify the sentiment of a product review."""
    review: str = dspy.InputField(desc="A product review from a customer")
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField(desc="The sentiment label")

# Program
classifier = dspy.ChainOfThought(ClassifySentiment)

# Training set
trainset = [
    dspy.Example(review="Absolutely love this product! Best purchase I've made all year.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Broke after two days. Complete waste of money.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="It works fine. Nothing special but gets the job done.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="The quality exceeded my expectations. Highly recommend!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Terrible customer service. Product arrived damaged.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="Average product. Decent for the price.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="This changed my life! Can't imagine going back.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Doesn't work as advertised. Very disappointed.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="It's okay. Not great, not terrible.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="Five stars! Everything I wanted and more.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Returned it the same day. Awful quality.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="Meets basic expectations. Would consider buying again.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="My whole family loves it. Ordering more as gifts!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Stopped working after a week. No response from support.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="Solid product. Does what it says.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="Incredible value for money. Blown away!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Flimsy and cheap. Not worth half the price.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="Pretty standard. Works as expected.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="The best in its category. Perfection!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Worst purchase I've ever made. Stay away.", sentiment="negative").with_inputs("review"),
]

devset = [
    dspy.Example(review="Great build quality and fast shipping!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Not what I expected. The description was misleading.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="It's a standard product. Nothing to complain about.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="Love the design and it works perfectly.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Arrived broken. Requesting a refund.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="Functional. Does the basics well enough.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="Outstanding! This is premium quality.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Poor materials. Feels like it'll break any moment.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="Middle of the road. Neither impressed nor disappointed.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="Exceeded all my expectations. A must-buy!", sentiment="positive").with_inputs("review"),
]

# Trace-aware metric: during bootstrapping, require that reasoning
# mentions the key sentiment signals from the review
def classify_metric(example, prediction, trace=None):
    correct = prediction.sentiment == example.sentiment
    if trace is not None:
        # During optimization: also require reasoning
        reasoning = getattr(prediction, "reasoning", "")
        has_reasoning = len(reasoning) > 20
        return correct and has_reasoning
    return correct

# Evaluate baseline
evaluator = Evaluate(
    devset=devset,
    metric=classify_metric,
    num_threads=4,
    display_progress=True,
    display_table=5,
)
baseline_score = evaluator(classifier)
print(f"Baseline: {baseline_score:.1f}%")

# Optimize — use only bootstrapped demos (no raw labeled demos)
# This forces all demos to include reasoning traces
optimizer = dspy.BootstrapFewShot(
    metric=classify_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=0,   # only bootstrapped demos with reasoning
)
optimized_classifier = optimizer.compile(classifier, trainset=trainset)

# Evaluate optimized program
optimized_score = evaluator(optimized_classifier)
print(f"Optimized: {optimized_score:.1f}%")
print(f"Delta: {optimized_score - baseline_score:+.1f}%")

# Save for production use
optimized_classifier.save("optimized_classifier.json")

# Load later
loaded = dspy.ChainOfThought(ClassifySentiment)
loaded.load("optimized_classifier.json")
result = loaded(review="This product is a game changer!")
print(f"Sentiment: {result.sentiment}")
```
