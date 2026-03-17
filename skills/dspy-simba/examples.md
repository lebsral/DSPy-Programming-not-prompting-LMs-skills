# dspy.SIMBA Examples

## Example 1: Incremental optimization of a classification pipeline

A support ticket classifier that already works at ~70% accuracy. SIMBA targets the hardest tickets -- ambiguous ones where the model is inconsistent -- and incrementally pushes accuracy higher.

```python
import dspy
from dspy.evaluate import Evaluate

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


# Define the classification signature
class ClassifyTicket(dspy.Signature):
    """Classify a support ticket into the correct department."""
    ticket_text: str = dspy.InputField(desc="The customer support ticket")
    department: str = dspy.OutputField(
        desc="One of: billing, technical, account, shipping, general"
    )


# Build the program
class TicketRouter(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(ClassifyTicket)

    def forward(self, ticket_text):
        return self.classify(ticket_text=ticket_text)


# Prepare training and dev data
raw_data = [
    ("I was charged twice for my subscription", "billing"),
    ("The app crashes when I open settings", "technical"),
    ("I need to update my email address", "account"),
    ("My package hasn't arrived in 2 weeks", "shipping"),
    ("How do I export my data?", "technical"),
    ("Can I get a refund for last month?", "billing"),
    ("I forgot my password and can't reset it", "account"),
    ("The tracking number shows delivered but I didn't get it", "shipping"),
    ("Your API returns 500 errors intermittently", "technical"),
    ("I want to cancel and get a prorated refund", "billing"),
    ("How do I add a team member to my account?", "account"),
    ("The delivery was left at the wrong address", "shipping"),
    ("Integration with Slack stopped working", "technical"),
    ("I see an unknown charge on my invoice", "billing"),
    ("Can I merge two accounts?", "account"),
    ("My order status hasn't updated in 5 days", "shipping"),
    ("How do I contact support?", "general"),
    ("What are your business hours?", "general"),
    ("The checkout page won't load on mobile", "technical"),
    ("I need a copy of my receipt from January", "billing"),
    ("Where is your return policy?", "general"),
    ("Dashboard loading times are very slow", "technical"),
    ("I want to downgrade my plan", "billing"),
    ("How do I enable two-factor authentication?", "account"),
    ("Package arrived damaged", "shipping"),
    ("Do you offer student discounts?", "general"),
    ("OAuth login fails with Google accounts", "technical"),
    ("I was billed after cancelling", "billing"),
    ("Can I transfer my subscription to someone else?", "account"),
    ("Shipment is stuck in customs", "shipping"),
    ("File upload feature is broken", "technical"),
    ("I need an itemized invoice for tax purposes", "billing"),
    ("How do I delete my account permanently?", "account"),
    ("Wrong item was shipped to me", "shipping"),
    ("What payment methods do you accept?", "general"),
    ("The search feature returns no results", "technical"),
    ("Autopay didn't process this month", "billing"),
    ("I can't change my username", "account"),
    ("Estimated delivery date keeps changing", "shipping"),
    ("Is there a desktop app?", "general"),
]

examples = [
    dspy.Example(ticket_text=text, department=dept).with_inputs("ticket_text")
    for text, dept in raw_data
]

trainset = examples[:30]
devset = examples[30:]


# Define the metric
def correct_department(example, prediction, trace=None):
    return prediction.department.lower().strip() == example.department.lower().strip()


# Evaluate baseline
program = TicketRouter()
evaluator = Evaluate(devset=devset, metric=correct_department, num_threads=4)
baseline_score = evaluator(program)
print(f"Baseline accuracy: {baseline_score:.1f}%")


# Step 1: Bootstrap a starting point
bootstrap = dspy.BootstrapFewShot(
    metric=correct_department,
    max_bootstrapped_demos=3,
)
bootstrapped = bootstrap.compile(program, trainset=trainset)

bootstrap_score = evaluator(bootstrapped)
print(f"After BootstrapFewShot: {bootstrap_score:.1f}%")


# Step 2: Incrementally improve with SIMBA
optimizer = dspy.SIMBA(
    metric=correct_department,
    bsize=16,           # smaller batches since dataset is small
    num_candidates=4,   # moderate exploration
    max_steps=6,        # enough iterations to find improvements
    max_demos=4,        # keep prompts manageable
)

optimized = optimizer.compile(bootstrapped, trainset=trainset)


# Evaluate the optimized program
final_score = evaluator(optimized)
print(f"After SIMBA: {final_score:.1f}%")


# Inspect what SIMBA found
print(f"\nCandidate programs found: {len(optimized.candidate_programs)}")
for i, (prog, score) in enumerate(optimized.candidate_programs[:5]):
    print(f"  Candidate {i}: avg_score={score:.3f}")


# Save the best program
optimized.save("optimized_ticket_router.json")
```

What this demonstrates:

- **Two-phase optimization** -- BootstrapFewShot establishes a baseline, then SIMBA incrementally improves it
- **Smaller `bsize`** for a small dataset -- 16 instead of the default 32 to keep mini-batches meaningful
- **Reduced `num_candidates` and `max_steps`** to match the dataset size and budget
- **Evaluation at each stage** to track improvement from baseline to bootstrapped to SIMBA-optimized
- **Inspecting candidate programs** to understand what alternatives SIMBA explored


## Example 2: Conservative tuning for production stability

A production Q&A system that must improve without regressing on already-correct answers. SIMBA's small-step approach is ideal here: each iteration makes a minimal change, and you can validate against a held-out set after each step.

```python
import dspy
from dspy.evaluate import Evaluate

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


# A multi-step Q&A pipeline already in production
class ProductionQA(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought(
            "question -> key_concepts"
        )
        self.answer = dspy.ChainOfThought(
            "question, key_concepts -> answer"
        )

    def forward(self, question):
        analysis = self.analyze(question=question)
        return self.answer(
            question=question,
            key_concepts=analysis.key_concepts,
        )


# Production dataset with known-good answers
qa_pairs = [
    ("What causes tides?", "gravitational pull of the moon and sun"),
    ("Why is the sky blue?", "rayleigh scattering of sunlight"),
    ("How do vaccines work?", "stimulate immune response with weakened or inactive pathogens"),
    ("What is photosynthesis?", "process where plants convert sunlight into energy"),
    ("Why do seasons change?", "earth's axial tilt relative to its orbit around the sun"),
    ("How does GPS work?", "triangulation using signals from multiple satellites"),
    ("What causes earthquakes?", "movement of tectonic plates"),
    ("How do antibiotics work?", "kill bacteria or stop them from reproducing"),
    ("What is inflation?", "general increase in prices and decrease in purchasing power"),
    ("How does WiFi work?", "radio waves transmitting data between devices and a router"),
    ("What causes thunder?", "rapid expansion of air heated by lightning"),
    ("How do magnets work?", "alignment of magnetic domains creating a magnetic field"),
    ("What is DNA?", "molecule carrying genetic instructions for development and function"),
    ("Why do we dream?", "brain processes memories and emotions during sleep"),
    ("How does a car engine work?", "internal combustion converts fuel into mechanical energy"),
    ("What causes rainbows?", "refraction and reflection of light in water droplets"),
    ("How do airplanes fly?", "lift generated by air pressure difference over wings"),
    ("What is machine learning?", "algorithms that improve through experience with data"),
    ("Why do leaves change color?", "chlorophyll breaks down revealing other pigments"),
    ("How does the internet work?", "network of networks using standardized protocols to route data"),
]

examples = [
    dspy.Example(question=q, answer=a).with_inputs("question")
    for q, a in qa_pairs
]

trainset = examples[:14]
devset = examples[14:]


# Graduated metric -- partial credit for close answers
def answer_quality(example, prediction, trace=None):
    pred = prediction.answer.lower().strip()
    gold = example.answer.lower().strip()

    # Exact match
    if gold in pred:
        return 1.0

    # Check for key term overlap
    gold_terms = set(gold.split())
    pred_terms = set(pred.split())
    overlap = gold_terms & pred_terms
    if not gold_terms:
        return 0.0

    overlap_ratio = len(overlap) / len(gold_terms)

    # Penalize very long answers (want conciseness)
    length_penalty = 1.0
    if len(pred.split()) > 50:
        length_penalty = 0.8

    return overlap_ratio * length_penalty


# Load the existing production program (or start fresh)
program = ProductionQA()

# Evaluate current production performance
evaluator = Evaluate(
    devset=devset,
    metric=answer_quality,
    num_threads=4,
    display_progress=True,
)
production_score = evaluator(program)
print(f"Current production score: {production_score:.2f}")


# Conservative SIMBA optimization
# - Low num_candidates to limit the size of changes
# - Default temperature to keep outputs stable
# - max_demos=3 to avoid bloating the production prompt
optimizer = dspy.SIMBA(
    metric=answer_quality,
    bsize=12,            # small batches from limited data
    num_candidates=4,    # conservative -- fewer candidates per step
    max_steps=6,         # moderate iteration count
    max_demos=3,         # keep prompts lean for production
)

optimized = optimizer.compile(program, trainset=trainset)

# Validate on held-out set
optimized_score = evaluator(optimized)
print(f"Optimized score: {optimized_score:.2f}")
print(f"Improvement: {optimized_score - production_score:+.2f}")


# Safety check: verify no regression on individual examples
print("\nPer-example comparison:")
regressions = 0
improvements = 0

for ex in devset:
    old_pred = program(question=ex.question)
    new_pred = optimized(question=ex.question)

    old_score = answer_quality(ex, old_pred)
    new_score = answer_quality(ex, new_pred)

    if new_score < old_score - 0.1:
        regressions += 1
        print(f"  REGRESSION: '{ex.question}' ({old_score:.2f} -> {new_score:.2f})")
    elif new_score > old_score + 0.1:
        improvements += 1
        print(f"  IMPROVED:   '{ex.question}' ({old_score:.2f} -> {new_score:.2f})")

print(f"\nImprovements: {improvements}, Regressions: {regressions}")


# Only deploy if no regressions (or regressions are acceptable)
if regressions == 0:
    optimized.save("production_qa_optimized.json")
    print("Safe to deploy -- saved optimized program.")
else:
    print(f"Found {regressions} regression(s) -- review before deploying.")
    # Still save for analysis
    optimized.save("production_qa_candidate.json")
```

What this demonstrates:

- **Production safety workflow** -- evaluates on a held-out dev set and checks for regressions before saving
- **Conservative parameters** -- low `num_candidates` (4) and `max_demos` (3) to minimize prompt changes
- **Graduated metric** -- partial credit for keyword overlap plus a length penalty for conciseness
- **Per-example regression analysis** -- compares old and new predictions individually to catch quality drops
- **Conditional deployment** -- only saves as the production model if no regressions are found
- **Multi-step pipeline** -- SIMBA optimizes both the `analyze` and `answer` predictors within `ProductionQA`
