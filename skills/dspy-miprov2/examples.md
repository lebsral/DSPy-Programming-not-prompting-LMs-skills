# MIPROv2 Examples

## Production Optimization with MIPROv2 auto="medium"

A sentiment classifier optimized with MIPROv2 for production use. Shows the full workflow: baseline evaluation, optimization, comparison, and saving the result.

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Program: sentiment classification with reasoning
class SentimentClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought("review -> sentiment: str")

    def forward(self, review):
        return self.classify(review=review)

classifier = SentimentClassifier()

# Training data (80% of your examples)
trainset = [
    dspy.Example(review="Absolutely love this product! Works perfectly.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Terrible quality. Broke after one day.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="It's okay. Nothing special but gets the job done.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="Best purchase I've made all year!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Waste of money. Do not buy.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="Decent for the price. Some minor issues.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="Five stars! Exceeded all my expectations.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Arrived damaged and customer support was unhelpful.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="Average product. Works as described.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="My family uses this every day. Highly recommend!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Stopped working after a week. Very disappointed.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="Not bad, not great. It's fine.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="This changed my morning routine for the better!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Poor build quality. Feels cheap.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="Does what it says. No complaints.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="Incredible value. Would buy again in a heartbeat.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Returned it the same day. Completely useless.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="Solid product for everyday use.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="A game changer! So glad I found this.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="The worst product I have ever purchased.", sentiment="negative").with_inputs("review"),
]

# Held-out dev set (20% of your examples)
devset = [
    dspy.Example(review="Outstanding quality and fast shipping!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Doesn't work as advertised. Frustrating.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="It's alright. Meets basic expectations.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="Pleasantly surprised by how well this works.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Cheap materials, fell apart quickly.", sentiment="negative").with_inputs("review"),
]

# Metric: normalized match on sentiment label
def sentiment_match(example, prediction, trace=None):
    pred = prediction.sentiment.strip().lower()
    gold = example.sentiment.strip().lower()
    match = pred == gold
    if trace is not None:
        # During optimization, also require reasoning
        has_reasoning = len(getattr(prediction, "reasoning", "")) > 20
        return match and has_reasoning
    return match

# Evaluate baseline
evaluator = Evaluate(
    devset=devset,
    metric=sentiment_match,
    num_threads=4,
    display_progress=True,
    display_table=5,
)
baseline_score = evaluator(classifier)
print(f"Baseline: {baseline_score:.1f}%")

# Optimize with MIPROv2
optimizer = dspy.MIPROv2(metric=sentiment_match, auto="medium")
optimized = optimizer.compile(classifier, trainset=trainset)

# Evaluate optimized program
optimized_score = evaluator(optimized)
print(f"Optimized: {optimized_score:.1f}%")
print(f"Delta:     {optimized_score - baseline_score:+.1f}%")

# Save for production
optimized.save("optimized_sentiment.json")

# Load later
production_classifier = SentimentClassifier()
production_classifier.load("optimized_sentiment.json")
result = production_classifier(review="This product is amazing!")
print(f"Sentiment: {result.sentiment}")
```

## Heavy Optimization for Maximum Quality

A multi-step RAG pipeline optimized with `auto="heavy"` for the highest quality. Demonstrates optimizing a complex program with multiple modules, using a composite metric, and stacking optimizers.

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Multi-step program: extract key info, then generate a detailed answer
class DetailedQA(dspy.Module):
    def __init__(self):
        self.extract = dspy.ChainOfThought("question -> key_concepts: list[str]")
        self.answer = dspy.ChainOfThought("question, key_concepts -> answer")

    def forward(self, question):
        extraction = self.extract(question=question)
        return self.answer(question=question, key_concepts=extraction.key_concepts)

qa = DetailedQA()

# Training data
trainset = [
    dspy.Example(
        question="What causes tides on Earth?",
        answer="Tides are primarily caused by the gravitational pull of the Moon on Earth's oceans, with the Sun also contributing. The Moon's gravity creates a bulge of water on the side of Earth facing it and another on the opposite side, resulting in high tides."
    ).with_inputs("question"),
    dspy.Example(
        question="How does a refrigerator work?",
        answer="A refrigerator works by circulating a refrigerant through a cycle of compression and expansion. The compressor pressurizes the refrigerant gas, which then condenses into a liquid releasing heat. The liquid evaporates inside the fridge absorbing heat, cooling the interior."
    ).with_inputs("question"),
    dspy.Example(
        question="Why do leaves change color in autumn?",
        answer="Leaves change color because shorter days trigger trees to stop producing chlorophyll, the green pigment. As chlorophyll breaks down, other pigments like carotenoids (yellow/orange) and anthocyanins (red/purple) become visible."
    ).with_inputs("question"),
    dspy.Example(
        question="What is the greenhouse effect?",
        answer="The greenhouse effect is the process where certain gases in Earth's atmosphere trap heat from the Sun. Solar radiation passes through the atmosphere and warms the surface, which then emits infrared radiation. Greenhouse gases absorb this radiation and re-emit it, warming the atmosphere."
    ).with_inputs("question"),
    dspy.Example(
        question="How do vaccines work?",
        answer="Vaccines introduce a weakened or inactive form of a pathogen to the immune system. This triggers the body to produce antibodies and memory cells without causing the disease. If exposed to the real pathogen later, the immune system can respond quickly."
    ).with_inputs("question"),
    dspy.Example(
        question="What causes earthquakes?",
        answer="Earthquakes are caused by the sudden release of energy in Earth's crust, usually due to tectonic plates moving past, colliding with, or pulling apart from each other. The point where the rupture starts is the focus, and the point directly above on the surface is the epicenter."
    ).with_inputs("question"),
    dspy.Example(
        question="How does GPS work?",
        answer="GPS works using a network of satellites orbiting Earth. A GPS receiver calculates its position by measuring the time signals take to arrive from at least four satellites. Using these time differences and the known positions of the satellites, it triangulates the receiver's location."
    ).with_inputs("question"),
    dspy.Example(
        question="Why is the sky blue?",
        answer="The sky appears blue because of Rayleigh scattering. Sunlight contains all colors, but shorter blue wavelengths scatter more when hitting gas molecules in the atmosphere. This scattered blue light reaches our eyes from all directions, making the sky look blue."
    ).with_inputs("question"),
    dspy.Example(
        question="How do antibiotics work?",
        answer="Antibiotics work by either killing bacteria or stopping them from reproducing. Some target the bacterial cell wall, others interfere with protein synthesis or DNA replication. They are effective against bacteria but not viruses."
    ).with_inputs("question"),
    dspy.Example(
        question="What causes rainbows?",
        answer="Rainbows form when sunlight enters water droplets, refracts (bends), reflects off the back of the droplet, and refracts again as it exits. This process separates white light into its component colors, creating the visible spectrum arc."
    ).with_inputs("question"),
]

# Held-out dev set
devset = [
    dspy.Example(
        question="How do solar panels generate electricity?",
        answer="Solar panels use photovoltaic cells made of semiconductor materials like silicon. When photons from sunlight hit the cells, they knock electrons loose, creating an electric current. This direct current is then converted to alternating current by an inverter."
    ).with_inputs("question"),
    dspy.Example(
        question="Why do we dream?",
        answer="The exact purpose of dreaming is debated, but leading theories suggest dreams help with memory consolidation, emotional processing, and problem-solving. During REM sleep, the brain is highly active and replays and reorganizes experiences from waking life."
    ).with_inputs("question"),
    dspy.Example(
        question="How does Wi-Fi work?",
        answer="Wi-Fi uses radio waves to transmit data between devices and a router. The router connects to the internet via a wired connection and broadcasts a wireless signal. Devices with Wi-Fi adapters send and receive data by modulating and demodulating these radio signals."
    ).with_inputs("question"),
]

# Composite metric: correctness + completeness + conciseness
class JudgeAnswer(dspy.Signature):
    """Judge whether the predicted answer correctly and completely covers the key facts in the reference answer."""
    question: str = dspy.InputField()
    reference_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    is_correct: bool = dspy.OutputField(desc="True if key facts are covered accurately")
    is_complete: bool = dspy.OutputField(desc="True if no major facts are missing")

judge_lm = dspy.LM("openai/gpt-4o")

def quality_metric(example, prediction, trace=None):
    # Correctness and completeness via LM judge
    with dspy.context(lm=judge_lm):
        judge = dspy.Predict(JudgeAnswer)
        result = judge(
            question=example.question,
            reference_answer=example.answer,
            predicted_answer=prediction.answer,
        )
    correct = float(result.is_correct)
    complete = float(result.is_complete)

    # Conciseness heuristic
    word_count = len(prediction.answer.split())
    if word_count <= 80:
        concise = 1.0
    elif word_count <= 150:
        concise = 0.5
    else:
        concise = 0.0

    # During optimization, require correctness and completeness
    if trace is not None:
        return correct and complete and concise >= 0.5

    return 0.5 * correct + 0.3 * complete + 0.2 * concise

# Evaluate baseline
evaluator = Evaluate(
    devset=devset,
    metric=quality_metric,
    num_threads=4,
    display_progress=True,
    display_table=3,
)
baseline_score = evaluator(qa)
print(f"Baseline: {baseline_score:.1f}%")

# Step 1: Quick bootstrap to find good demos
bootstrap = dspy.BootstrapFewShot(metric=quality_metric, max_bootstrapped_demos=4)
bootstrapped = bootstrap.compile(qa, trainset=trainset)

bootstrap_score = evaluator(bootstrapped)
print(f"After bootstrap: {bootstrap_score:.1f}%")

# Step 2: Heavy MIPROv2 optimization on the bootstrapped result
optimizer = dspy.MIPROv2(metric=quality_metric, auto="heavy")
final = optimizer.compile(bootstrapped, trainset=trainset)

# Evaluate final result
final_score = evaluator(final)
print(f"\nResults:")
print(f"  Baseline:        {baseline_score:.1f}%")
print(f"  After bootstrap: {bootstrap_score:.1f}%")
print(f"  After MIPROv2:   {final_score:.1f}%")
print(f"  Total delta:     {final_score - baseline_score:+.1f}%")

# Save the final optimized program
final.save("optimized_detailed_qa.json")
```
