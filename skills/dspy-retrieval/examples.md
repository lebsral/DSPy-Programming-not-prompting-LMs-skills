# dspy-retrieval -- Worked Examples

## Example 1: RAG pipeline with ColBERTv2

A complete RAG pipeline that uses a hosted ColBERTv2 server to retrieve Wikipedia passages and answer questions with citations.

```python
import dspy
from typing import Literal


class AnswerWithConfidence(dspy.Signature):
    """Answer the question using only the provided context. Say 'insufficient information' if the context doesn't contain the answer."""
    context: list[str] = dspy.InputField(desc="Retrieved passages from the knowledge base")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Answer grounded in the context")
    confidence: Literal["high", "medium", "low"] = dspy.OutputField()


class ColBERTRAG(dspy.Module):
    """RAG pipeline backed by ColBERTv2 retrieval."""

    def __init__(self, k=5):
        self.retrieve = dspy.Retrieve(k=k)
        self.generate = dspy.ChainOfThought(AnswerWithConfidence)

    def forward(self, question):
        # Retrieve relevant passages
        passages = self.retrieve(question).passages

        if not passages:
            return dspy.Prediction(
                answer="No relevant passages found.",
                confidence="low",
                passages=[],
            )

        # Generate a grounded answer
        result = self.generate(context=passages, question=question)

        return dspy.Prediction(
            answer=result.answer,
            confidence=result.confidence,
            passages=passages,
        )


def rag_confidence_reward(args, pred):
    """Prefer high- or medium-confidence answers; penalize low confidence."""
    if pred.confidence == "low":
        return 0.5
    return 1.0


# --- Setup ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=lm, rm=colbert)

# --- Usage ---

rag = dspy.Refine(module=ColBERTRAG(k=5), N=3, reward_fn=rag_confidence_reward, threshold=1.0)
result = rag(question="What is the capital of France?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Retrieved {len(result.passages)} passages")

# --- Evaluation ---

devset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="Who wrote Hamlet?", answer="William Shakespeare").with_inputs("question"),
]


def rag_metric(example, prediction, trace=None):
    answer_correct = example.answer.lower() in prediction.answer.lower()
    high_confidence = prediction.confidence in ("high", "medium")
    return answer_correct + 0.3 * high_confidence


from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=devset, metric=rag_metric, num_threads=2)
score = evaluator(rag)
print(f"Score: {score}")

# --- Optimization ---

optimizer = dspy.BootstrapFewShot(metric=rag_metric, max_bootstrapped_demos=4)
optimized_rag = optimizer.compile(rag, trainset=devset)
optimized_rag.save("colbert_rag_optimized.json")
```

Key points:
- `dspy.ColBERTv2` is set as the global retrieval model via `dspy.configure(rm=colbert)`
- `dspy.Retrieve(k=5)` delegates to ColBERTv2 automatically
- The module handles the empty-results edge case before calling the LM
- `dspy.Refine` retries generation when confidence is low, nudging the model toward higher-quality answers
- Optimization tunes the answer generation prompt while leaving retrieval unchanged


## Example 2: Custom retriever with embeddings

Build a local retriever using `dspy.Embedder` and `dspy.retrievers.Embeddings` over your own document corpus. No external server needed.

```python
import dspy


# --- Prepare corpus ---

documents = [
    "DSPy is a framework for programming language models instead of prompting them.",
    "Retrieval-augmented generation (RAG) combines search with language model generation.",
    "ColBERTv2 is a neural retrieval model that uses late interaction for efficient passage ranking.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Few-shot learning uses a small number of examples to teach a model a new task.",
    "Chain-of-thought prompting improves reasoning by generating intermediate steps.",
    "Vector databases store embeddings for fast nearest-neighbor search.",
    "DSPy optimizers tune prompts automatically using training examples and a metric.",
    "Embeddings map text to dense vectors where similar texts are close together.",
    "Multi-hop retrieval chains multiple search steps to answer complex questions.",
]


# --- Build retriever ---

embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=documents, k=3)


# --- Build RAG module ---

class EmbeddingsRAG(dspy.Module):
    """RAG using local embeddings-based retrieval."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retriever(question).passages
        return self.answer(context=context, question=question)


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

rag = EmbeddingsRAG(retriever=search)
result = rag(question="What is DSPy?")
print(result.answer)
print(result.reasoning)

# --- Inspect what was retrieved ---

retrieved = search("What is DSPy?")
for i, passage in enumerate(retrieved.passages):
    print(f"  [{i}] {passage}")
```

Key points:
- `dspy.Embedder` handles embedding computation via LiteLLM -- works with OpenAI, Cohere, Ollama, etc.
- `dspy.retrievers.Embeddings` builds a FAISS index in memory from the corpus
- No global `rm` configuration needed -- the retriever is passed directly to the module
- The corpus is a plain `list[str]` -- load your documents however you like
- For larger corpora, consider using a dedicated vector store (Chroma, Pinecone) instead


## Example 3: Multi-hop retrieval pattern

Answer complex questions that require combining information from multiple documents. Each hop generates a new search query based on what has been found so far.

```python
import dspy


class GenerateSearchQuery(dspy.Signature):
    """Generate a search query to find information that is still missing."""
    context: list[str] = dspy.InputField(desc="Information gathered so far")
    question: str = dspy.InputField(desc="The original question to answer")
    search_query: str = dspy.OutputField(desc="A focused search query for the next retrieval step")


class AnswerFromContext(dspy.Signature):
    """Answer the question using all gathered context. Cite specific facts from the passages."""
    context: list[str] = dspy.InputField(desc="All retrieved passages across search steps")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Comprehensive answer grounded in the context")


class MultiHopRAG(dspy.Module):
    """Multi-hop retrieval: iteratively search, gather context, then answer."""

    def __init__(self, retriever, hops=2, passages_per_hop=3):
        self.retriever = retriever
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(hops)
        ]
        self.answer = dspy.ChainOfThought(AnswerFromContext)

    def forward(self, question):
        context = []

        for hop in self.generate_query:
            # Generate a search query based on what we know so far
            search_query = hop(context=context, question=question).search_query

            # Retrieve new passages
            new_passages = self.retriever(search_query).passages

            # Deduplicate and accumulate context
            context = list(dict.fromkeys(context + new_passages))

        # Generate final answer from all gathered context
        result = self.answer(context=context, question=question)

        return dspy.Prediction(
            answer=result.answer,
            reasoning=result.reasoning,
            context=context,
            num_hops=len(self.generate_query),
        )


# --- Setup with Embeddings retriever ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

corpus = [
    "The Eiffel Tower is located in Paris, France. It was built in 1889.",
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower was designed by Gustave Eiffel's engineering company.",
    "Gustave Eiffel was born on December 15, 1832 in Dijon, France.",
    "The Eiffel Tower stands 330 meters tall and was the tallest structure in the world until 1930.",
    "France is a country in Western Europe with a population of about 67 million.",
    "The Chrysler Building surpassed the Eiffel Tower as the tallest structure in 1930.",
    "Dijon is a city in eastern France, known as the capital of the Burgundy region.",
]

embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=3)


def multihop_context_reward(args, pred):
    """Reward multi-hop results that gathered passages from multiple search steps."""
    if len(pred.context) < 2:
        return 0.5
    return 1.0


# --- Usage ---

multihop = dspy.Refine(
    module=MultiHopRAG(retriever=search, hops=2, passages_per_hop=3),
    N=3,
    reward_fn=multihop_context_reward,
    threshold=1.0,
)
result = multihop(question="Where was the designer of the Eiffel Tower born?")

print(f"Answer: {result.answer}")
print(f"Reasoning: {result.reasoning}")
print(f"Hops: {result.num_hops}")
print(f"Context gathered ({len(result.context)} passages):")
for i, passage in enumerate(result.context):
    print(f"  [{i}] {passage}")


# --- Evaluation ---

devset = [
    dspy.Example(
        question="Where was the designer of the Eiffel Tower born?",
        answer="Dijon, France",
    ).with_inputs("question"),
    dspy.Example(
        question="What surpassed the Eiffel Tower as the tallest structure?",
        answer="The Chrysler Building",
    ).with_inputs("question"),
]


def multihop_metric(example, prediction, trace=None):
    answer_correct = example.answer.lower() in prediction.answer.lower()
    used_multiple_passages = len(prediction.context) > 3
    return answer_correct + 0.2 * used_multiple_passages


# --- Optimization ---

# optimizer = dspy.BootstrapFewShot(metric=multihop_metric, max_bootstrapped_demos=3)
# optimized = optimizer.compile(multihop, trainset=devset)
# optimized.save("multihop_rag_optimized.json")
```

Key points:
- Each hop generates a new search query using `ChainOfThought`, which reasons about what information is still missing
- The `generate_query` list creates separate `ChainOfThought` instances per hop -- each gets its own optimizable prompts
- `dict.fromkeys()` deduplicates passages while preserving order
- Multi-hop is essential for compositional questions like "Where was the designer of X born?" where no single passage has the full answer
- The optimizer tunes both query generation and answer generation together, improving end-to-end accuracy
