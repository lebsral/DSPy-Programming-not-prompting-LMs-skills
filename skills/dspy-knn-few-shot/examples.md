# dspy-knn-few-shot -- Worked Examples

## Example 1: Dynamic demo selection for classification

Classify support tickets into categories. With 8+ categories and varied phrasing, static few-shot demos often include irrelevant examples. KNNFewShot retrieves the most relevant tickets for each new input.

```python
import dspy
from typing import Literal
from sentence_transformers import SentenceTransformer

# --- Setup ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

encoder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = dspy.Embedder(encoder.encode)


# --- Signature ---

class ClassifyTicket(dspy.Signature):
    """Classify a support ticket into the correct category."""
    ticket: str = dspy.InputField(desc="The support ticket text")
    category: Literal[
        "billing", "bug_report", "feature_request", "account_access",
        "performance", "documentation", "integration", "general"
    ] = dspy.OutputField(desc="The ticket category")


# --- Training data ---
# In practice you'd load these from a database or CSV.

trainset = [
    dspy.Example(ticket="I was charged twice for my subscription this month", category="billing").with_inputs("ticket"),
    dspy.Example(ticket="Can you refund my last payment?", category="billing").with_inputs("ticket"),
    dspy.Example(ticket="The dashboard crashes when I click on reports", category="bug_report").with_inputs("ticket"),
    dspy.Example(ticket="Getting a 500 error on the API endpoint /users", category="bug_report").with_inputs("ticket"),
    dspy.Example(ticket="It would be great to have dark mode", category="feature_request").with_inputs("ticket"),
    dspy.Example(ticket="Can you add support for exporting to PDF?", category="feature_request").with_inputs("ticket"),
    dspy.Example(ticket="I can't log in, it says my password is wrong", category="account_access").with_inputs("ticket"),
    dspy.Example(ticket="My account was locked after too many attempts", category="account_access").with_inputs("ticket"),
    dspy.Example(ticket="The page takes 30 seconds to load", category="performance").with_inputs("ticket"),
    dspy.Example(ticket="API response times are very slow today", category="performance").with_inputs("ticket"),
    dspy.Example(ticket="The docs for webhooks are outdated", category="documentation").with_inputs("ticket"),
    dspy.Example(ticket="I can't find any docs on the new batch API", category="documentation").with_inputs("ticket"),
    dspy.Example(ticket="How do I connect Slack to your platform?", category="integration").with_inputs("ticket"),
    dspy.Example(ticket="The Salesforce sync stopped working after the update", category="integration").with_inputs("ticket"),
    dspy.Example(ticket="What are your support hours?", category="general").with_inputs("ticket"),
    dspy.Example(ticket="Do you have an office in Europe?", category="general").with_inputs("ticket"),
]

# --- Compile with KNNFewShot ---

knn_optimizer = dspy.KNNFewShot(
    k=3,
    trainset=trainset,
    vectorizer=embedder,
)

classifier = dspy.Predict(ClassifyTicket)
optimized_classifier = knn_optimizer.compile(classifier)

# --- Use it ---

# A billing-related ticket: the 3 demos will be the most billing-like examples
result = optimized_classifier(ticket="Why was I charged $49.99 instead of $29.99?")
print(result.category)  # billing

# A bug-report ticket: demos shift to the most bug-like examples
result = optimized_classifier(ticket="The export button doesn't work on Safari")
print(result.category)  # bug_report

# An integration ticket: demos shift to integration-related examples
result = optimized_classifier(ticket="Can I use your API with Zapier?")
print(result.category)  # integration
```

Key points:
- Each call gets different demos based on the ticket content. A billing question sees billing examples; a bug report sees bug examples.
- `k=3` keeps the prompt short while giving the LM enough context to distinguish categories.
- `dspy.Predict` is used instead of `ChainOfThought` because classification is straightforward -- no reasoning chain needed.
- With 8 categories and only 3 demo slots, static few-shot would miss most categories for any given input. KNN ensures the right categories are always represented.
- The training set has 2 examples per category (16 total). In production, 5-10 per category gives better retrieval coverage.


## Example 2: KNNFewShot with custom embeddings

Use OpenAI embeddings instead of sentence-transformers, and pass BootstrapFewShot arguments to control demo generation. This example shows a QA task where the training data includes both questions and gold answers.

```python
import dspy
import openai
from dspy.evaluate import Evaluate

# --- Setup ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# --- Custom embedder using OpenAI ---

client = openai.OpenAI()

def openai_embed(texts):
    """Embed texts using OpenAI's embedding API."""
    # Handle both single string and list of strings
    if isinstance(texts, str):
        texts = [texts]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]

embedder = dspy.Embedder(openai_embed)


# --- Signature ---

class AnswerQuestion(dspy.Signature):
    """Answer a factual question concisely."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="A concise factual answer")


# --- Training and dev data ---

trainset = [
    dspy.Example(question="What is the speed of light?", answer="approximately 299,792 km/s").with_inputs("question"),
    dspy.Example(question="What is the boiling point of water?", answer="100 degrees Celsius at sea level").with_inputs("question"),
    dspy.Example(question="Who wrote Romeo and Juliet?", answer="William Shakespeare").with_inputs("question"),
    dspy.Example(question="What is the largest planet in our solar system?", answer="Jupiter").with_inputs("question"),
    dspy.Example(question="What year did World War II end?", answer="1945").with_inputs("question"),
    dspy.Example(question="What is the chemical symbol for gold?", answer="Au").with_inputs("question"),
    dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
    dspy.Example(question="What is the tallest mountain on Earth?", answer="Mount Everest").with_inputs("question"),
    dspy.Example(question="What is the atomic number of carbon?", answer="6").with_inputs("question"),
    dspy.Example(question="What language has the most native speakers?", answer="Mandarin Chinese").with_inputs("question"),
    dspy.Example(question="What is the smallest country by area?", answer="Vatican City").with_inputs("question"),
    dspy.Example(question="What is the freezing point of water in Fahrenheit?", answer="32 degrees Fahrenheit").with_inputs("question"),
    dspy.Example(question="Who discovered penicillin?", answer="Alexander Fleming").with_inputs("question"),
    dspy.Example(question="What is the capital of Japan?", answer="Tokyo").with_inputs("question"),
    dspy.Example(question="What element does O represent on the periodic table?", answer="Oxygen").with_inputs("question"),
]

devset = [
    dspy.Example(question="What is the melting point of iron?", answer="1538 degrees Celsius").with_inputs("question"),
    dspy.Example(question="Who wrote Pride and Prejudice?", answer="Jane Austen").with_inputs("question"),
    dspy.Example(question="What is the chemical symbol for silver?", answer="Ag").with_inputs("question"),
    dspy.Example(question="What is the second largest planet?", answer="Saturn").with_inputs("question"),
]


# --- Metric ---

def answer_match(example, pred, trace=None):
    """Check if the predicted answer contains the key information."""
    gold = example.answer.lower()
    predicted = pred.answer.lower()
    # Accept if the gold answer appears within the prediction
    return gold in predicted or predicted in gold


# --- Compile with KNNFewShot + BootstrapFewShot args ---

knn_optimizer = dspy.KNNFewShot(
    k=5,
    trainset=trainset,
    vectorizer=embedder,
    # These are forwarded to BootstrapFewShot:
    metric=answer_match,
    max_bootstrapped_demos=2,
    max_labeled_demos=3,
)

qa = dspy.ChainOfThought(AnswerQuestion)
optimized_qa = knn_optimizer.compile(qa)


# --- Evaluate ---

evaluator = Evaluate(
    devset=devset,
    metric=answer_match,
    num_threads=4,
    display_progress=True,
    display_table=5,
)

score = evaluator(optimized_qa)
print(f"Accuracy: {score}%")


# --- Use it ---

# Science question: retrieves science-related demos
result = optimized_qa(question="What is the density of water?")
print(result.answer)

# History question: retrieves history/literature demos
result = optimized_qa(question="Who invented the telephone?")
print(result.answer)
```

Key points:
- `dspy.Embedder(openai_embed)` wraps a custom OpenAI embedding function. The function must accept a string or list of strings and return a list of vectors.
- `k=5` retrieves 5 neighbors, then BootstrapFewShot selects up to 2 bootstrapped + 3 labeled demos from those 5. This two-stage filtering gives you the best of both worlds: relevant retrieval plus quality-based demo selection.
- The `metric`, `max_bootstrapped_demos`, and `max_labeled_demos` arguments are forwarded directly to BootstrapFewShot via `**few_shot_bootstrap_args`.
- OpenAI embeddings cost money per call (embedding happens at init for the trainset and at each query). For cost-sensitive workloads, sentence-transformers run locally for free.
- The `text-embedding-3-small` model produces normalized vectors, so dot-product similarity works correctly out of the box.
