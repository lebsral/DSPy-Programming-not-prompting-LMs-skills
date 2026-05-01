# dspy-bootstrap-rs -- Worked Examples

## Example 1: QA optimization with random search

Optimize a question-answering module by searching over multiple candidate demo sets. This shows the basic end-to-end workflow: prepare data, define a metric, run the optimizer, and compare against baseline.

```python
import dspy
from dspy.evaluate import Evaluate

# --- Setup ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# --- Data ---

dataset = [
    ("What language is DSPy written in?", "Python"),
    ("Who developed the transformer architecture?", "Google"),
    ("What does LLM stand for?", "Large Language Model"),
    ("What year was GPT-3 released?", "2020"),
    ("What framework does DSPy build on for LM calls?", "LiteLLM"),
    ("What is retrieval-augmented generation?", "Combining retrieval with generation"),
    ("What is the purpose of few-shot prompting?", "Providing examples to guide the model"),
    ("What does RLHF stand for?", "Reinforcement Learning from Human Feedback"),
    ("What is a token in NLP?", "A unit of text processed by the model"),
    ("What is prompt engineering?", "Designing inputs to get better LM outputs"),
    ("What is chain-of-thought prompting?", "Asking the model to show reasoning steps"),
    ("What is fine-tuning?", "Training a pre-trained model on task-specific data"),
    ("What does zero-shot mean?", "Performing a task without any examples"),
    ("What is an embedding?", "A dense vector representation of text"),
    ("What is attention in transformers?", "A mechanism for weighing token relevance"),
    ("What is beam search?", "A decoding strategy that keeps top-k candidates"),
    ("What is temperature in LM sampling?", "A parameter controlling output randomness"),
    ("What is a system prompt?", "Instructions that set the LM's behavior"),
    ("What is grounding in AI?", "Connecting model outputs to factual sources"),
    ("What does RAG stand for?", "Retrieval-Augmented Generation"),
]

examples = [
    dspy.Example(question=q, answer=a).with_inputs("question")
    for q, a in dataset
]

# Split into train and dev sets
trainset = examples[:15]
devset = examples[15:]


# --- Metric ---

def answer_match(example, prediction, trace=None):
    """Check if the predicted answer matches the gold answer (case-insensitive)."""
    pred = prediction.answer.strip().lower()
    gold = example.answer.strip().lower()
    # Exact match or gold answer is contained in the prediction
    return gold in pred


# --- Baseline ---

qa = dspy.ChainOfThought("question -> answer")

evaluator = Evaluate(
    devset=devset,
    metric=answer_match,
    num_threads=4,
    display_progress=True,
    display_table=5,
)

baseline_score = evaluator(qa)
print(f"Baseline score: {baseline_score:.1f}%")


# --- Optimize with BootstrapFewShotWithRandomSearch ---

optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=answer_match,
    max_bootstrapped_demos=4,     # Up to 4 program-generated demos
    max_labeled_demos=4,          # Up to 4 labeled demos from trainset
    num_candidate_programs=10,    # Try 10 different random demo sets
    num_threads=4,
)

optimized_qa = optimizer.compile(qa, trainset=trainset)


# --- Evaluate optimized program ---

optimized_score = evaluator(optimized_qa)
print(f"Baseline:  {baseline_score:.1f}%")
print(f"Optimized: {optimized_score:.1f}%")
print(f"Improvement: {optimized_score - baseline_score:+.1f}%")


# --- Save ---

optimized_qa.save("optimized_qa.json")
```

Key points:
- The optimizer tries 10 different random subsets of demos (`num_candidate_programs=10`) and picks the best-performing one
- The metric uses containment (`gold in pred`) rather than strict equality to handle minor formatting differences -- a common practical choice for QA tasks
- Training and dev sets are split so the optimizer doesn't overfit to the evaluation data
- Start with `num_candidate_programs=10` for a quick run; increase to 16-25 for more thorough search


## Example 2: Multi-step pipeline optimization

Optimize a two-stage pipeline (extract key facts, then generate an answer) where each stage has its own predictor that gets its own set of demos. BootstrapRS finds demos for all predictors in the pipeline simultaneously.

```python
import dspy
from dspy.evaluate import Evaluate


# --- Define a multi-step pipeline ---

class FactThenAnswer(dspy.Module):
    """Two-step QA: first extract relevant facts, then answer based on facts."""

    def __init__(self):
        self.extract_facts = dspy.ChainOfThought(
            "context, question -> key_facts: str"
        )
        self.answer = dspy.ChainOfThought(
            "key_facts, question -> answer: str"
        )

    def forward(self, context, question):
        facts = self.extract_facts(context=context, question=question)
        return self.answer(key_facts=facts.key_facts, question=question)


# --- Setup ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)


# --- Data ---

trainset = [
    dspy.Example(
        context="The Eiffel Tower was built in 1889 for the World's Fair. It stands 330 meters tall and is located in Paris, France. Gustave Eiffel's company designed and built it.",
        question="How tall is the Eiffel Tower?",
        answer="330 meters",
    ).with_inputs("context", "question"),
    dspy.Example(
        context="Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability and supports multiple programming paradigms.",
        question="Who created Python?",
        answer="Guido van Rossum",
    ).with_inputs("context", "question"),
    dspy.Example(
        context="The human genome contains approximately 3 billion base pairs of DNA. The Human Genome Project was completed in 2003 after 13 years of work.",
        question="How many base pairs are in the human genome?",
        answer="approximately 3 billion",
    ).with_inputs("context", "question"),
    dspy.Example(
        context="Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning. Elon Musk joined as chairman in 2004 and became CEO in 2008.",
        question="When was Tesla founded?",
        answer="2003",
    ).with_inputs("context", "question"),
    dspy.Example(
        context="The speed of light in a vacuum is 299,792,458 meters per second. Einstein's theory of special relativity states nothing can travel faster than light.",
        question="What is the speed of light?",
        answer="299,792,458 meters per second",
    ).with_inputs("context", "question"),
    dspy.Example(
        context="Water boils at 100 degrees Celsius at standard atmospheric pressure. At higher altitudes, the boiling point decreases due to lower air pressure.",
        question="At what temperature does water boil at standard pressure?",
        answer="100 degrees Celsius",
    ).with_inputs("context", "question"),
    dspy.Example(
        context="The Amazon River is about 6,400 km long, making it the second longest river in the world after the Nile. It flows through South America.",
        question="How long is the Amazon River?",
        answer="about 6,400 km",
    ).with_inputs("context", "question"),
    dspy.Example(
        context="Mount Everest is 8,849 meters above sea level, making it the highest peak on Earth. It is located in the Himalayas on the border of Nepal and Tibet.",
        question="How high is Mount Everest?",
        answer="8,849 meters",
    ).with_inputs("context", "question"),
    dspy.Example(
        context="The Great Wall of China stretches over 21,000 km. Construction began in the 7th century BC, and the most well-known sections were built during the Ming Dynasty.",
        question="How long is the Great Wall of China?",
        answer="over 21,000 km",
    ).with_inputs("context", "question"),
    dspy.Example(
        context="Jupiter is the largest planet in our solar system with a diameter of 139,820 km. It has at least 95 known moons, including the four large Galilean moons.",
        question="How many known moons does Jupiter have?",
        answer="at least 95",
    ).with_inputs("context", "question"),
]

devset = trainset[7:]   # last 3 for evaluation
trainset = trainset[:7] # first 7 for training


# --- Metric ---

def answer_match(example, prediction, trace=None):
    """Check if the gold answer is contained in the prediction."""
    pred = prediction.answer.strip().lower()
    gold = example.answer.strip().lower()
    return gold in pred


# --- Baseline ---

pipeline = FactThenAnswer()

evaluator = Evaluate(
    devset=devset,
    metric=answer_match,
    num_threads=4,
    display_progress=True,
    display_table=5,
)

baseline_score = evaluator(pipeline)
print(f"Baseline: {baseline_score:.1f}%")


# --- Optimize both stages with BootstrapRS ---

optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=answer_match,
    max_bootstrapped_demos=2,     # Fewer demos per stage to keep prompts manageable
    max_labeled_demos=2,
    num_candidate_programs=16,    # Search over 16 candidate demo combinations
    num_threads=4,
)

optimized_pipeline = optimizer.compile(pipeline, trainset=trainset)


# --- Evaluate ---

optimized_score = evaluator(optimized_pipeline)
print(f"Baseline:  {baseline_score:.1f}%")
print(f"Optimized: {optimized_score:.1f}%")
print(f"Improvement: {optimized_score - baseline_score:+.1f}%")


# --- Inspect what the optimizer chose ---

# Each predictor in the pipeline gets its own demos
print("\n--- Extract Facts demos ---")
for demo in optimized_pipeline.extract_facts.demos:
    print(f"  Q: {demo.get('question', 'N/A')[:60]}...")

print("\n--- Answer demos ---")
for demo in optimized_pipeline.answer.demos:
    print(f"  Q: {demo.get('question', 'N/A')[:60]}...")


# --- Save ---

optimized_pipeline.save("optimized_pipeline.json")
```

Key points:
- The pipeline has two predictors (`extract_facts` and `answer`), and the optimizer finds demos for both simultaneously -- each predictor gets its own demo set
- `max_bootstrapped_demos=2` and `max_labeled_demos=2` are lower than the single-module example because each stage adds demos to the prompt, and a two-stage pipeline needs to fit within context limits
- Bootstrapped demos are especially valuable here because the `extract_facts` step produces intermediate `key_facts` that only exist in successful traces -- labeled data alone wouldn't include these
- The optimizer evaluates end-to-end: a candidate is scored by how well the full pipeline's final answer matches, not by how well individual stages perform
- With `num_candidate_programs=16`, the optimizer tries 16 different combinations of demos across both stages and picks the combination that yields the best end-to-end score
