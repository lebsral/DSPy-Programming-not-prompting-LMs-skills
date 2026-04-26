---
name: dspy-retrieval
description: Use when you need to search over documents or build RAG pipelines — setting up retrievers, computing embeddings, or integrating vector databases with DSPy modules. Common scenarios: building RAG pipelines, connecting DSPy to a vector database, computing embeddings for semantic search, setting up ChromaDB or Pinecone with DSPy, retrieving relevant documents before generating answers, or building knowledge-grounded question answering. Related: ai-searching-docs, ai-stopping-hallucinations, dspy-modules. Also: dspy.Retrieve, dspy.ColBERTv2, RAG pipeline in DSPy, vector database integration, semantic search DSPy, ChromaDB with DSPy, Pinecone with DSPy, Weaviate with DSPy, embedding retrieval, document retrieval for QA, retrieval augmented generation setup, connect knowledge base to DSPy, search documents then answer, grounded generation with retrieval.
---

# Retrieval Modules in DSPy

Guide the user through DSPy's retrieval modules for searching documents, computing embeddings, and building RAG (retrieval-augmented generation) pipelines.

## What retrieval modules are

DSPy provides retrieval modules that fetch relevant documents or passages given a query. These modules plug into DSPy programs just like `dspy.Predict` or `dspy.ChainOfThought` -- declare them in `__init__`, call them in `forward()`, and optimizers handle the rest.

There are four key components:

| Component | Purpose | When to use |
|-----------|---------|-------------|
| `dspy.Retrieve` | Base retriever class | Wrap any search backend (Elastic, Pinecone, etc.) |
| `dspy.ColBERTv2` | ColBERTv2 retrieval client | Query a hosted ColBERTv2 server |
| `dspy.Embedder` | Compute embeddings | Turn text into vectors using any LiteLLM-supported model |
| `dspy.retrievers.Embeddings` | Local vector search | Build a retriever from an embedder + corpus, uses FAISS |

## dspy.Retrieve

The base class for all retrievers. Use it directly with a configured retrieval model (`rm`), or subclass it to wrap your own search backend.

### Using with a configured RM

```python
import dspy

# Configure a retrieval model globally
colbert = dspy.ColBERTv2(url="http://your-server:8893/api/search")
dspy.configure(lm=lm, rm=colbert)

# Use dspy.Retrieve -- it delegates to the configured rm
retriever = dspy.Retrieve(k=5)
result = retriever("What is retrieval-augmented generation?")
print(result.passages)  # list[str] of top-k passages
```

### Key parameters

- **`k`** (int) -- number of passages to retrieve. Can be set at init time or overridden per call.

### Return value

`dspy.Retrieve` returns a `dspy.Prediction` with a `.passages` attribute -- a `list[str]` of the top-k retrieved passages.

### Subclassing for custom backends

Wrap any search system by subclassing `dspy.Retrieve` and implementing `forward()`:

```python
class MyRetriever(dspy.Retrieve):
    def __init__(self, search_client, k=3):
        super().__init__(k=k)
        self.client = search_client

    def forward(self, query, k=None):
        k = k or self.k
        results = self.client.search(query, top_k=k)
        return dspy.Prediction(passages=[r["text"] for r in results])
```

The `forward()` method must:
1. Accept `query` (str) and optional `k` (int)
2. Return a `dspy.Prediction` with a `passages` field (list of strings)

## dspy.ColBERTv2

A retrieval client that queries a hosted ColBERTv2 server. ColBERTv2 is a neural retrieval model that provides high-quality passage retrieval.

### Constructor

```python
colbert = dspy.ColBERTv2(url="http://your-server:8893/api/search")
```

**Parameters:**
- **`url`** (str) -- URL of the ColBERTv2 server endpoint

### Usage

```python
# Direct call
results = colbert("What is DSPy?", k=3)
# Returns list of dicts with 'text', 'score', etc.

# As a configured retrieval model
dspy.configure(lm=lm, rm=colbert)
retriever = dspy.Retrieve(k=5)
passages = retriever("search query").passages
```

### Setting up a ColBERTv2 server

Stanford hosts a public ColBERTv2 server for Wikipedia that you can use for testing:

```python
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=lm, rm=colbert)
```

For your own data, you need to run a ColBERTv2 server. See the [ColBERT repository](https://github.com/stanford-futuredata/ColBERT) for setup instructions.

## dspy.Embedder

Computes embeddings for text using any LiteLLM-supported embedding model. This is not a retriever itself -- it turns text into vectors that you can use with `dspy.retrievers.Embeddings` or your own vector store.

### Constructor

```python
embedder = dspy.Embedder(
    "openai/text-embedding-3-small",  # model identifier (LiteLLM format)
    dimensions=512,                    # optional: output dimensions
)
```

**Parameters:**
- **model** (str) -- embedding model in LiteLLM format (e.g., `"openai/text-embedding-3-small"`, `"cohere/embed-english-v3.0"`)
- **dimensions** (int, optional) -- output embedding dimensions (if the model supports it)
- **batch_size** (int, optional) -- batch size for embedding multiple texts

### Usage

```python
# Embed a single text
vector = embedder("What is DSPy?")
# Returns a list of floats

# Embed multiple texts
vectors = embedder(["text one", "text two", "text three"])
# Returns a list of lists of floats
```

### Supported providers

Any embedding model supported by LiteLLM works:

```python
# OpenAI
embedder = dspy.Embedder("openai/text-embedding-3-small")

# Cohere
embedder = dspy.Embedder("cohere/embed-english-v3.0")

# Local via Ollama
embedder = dspy.Embedder("ollama/nomic-embed-text")
```

## dspy.retrievers.Embeddings

A local vector search retriever that uses FAISS under the hood. Give it an `Embedder` and a corpus, and it builds an in-memory index for fast similarity search.

### Constructor

```python
import dspy

embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
search = dspy.retrievers.Embeddings(
    embedder=embedder,
    corpus=corpus,     # list[str] of documents
    k=5,               # number of results to return
)
```

**Parameters:**
- **`embedder`** -- a `dspy.Embedder` instance
- **`corpus`** (list[str]) -- the documents to index and search over
- **`k`** (int) -- default number of results to return

### Usage

```python
# Search
result = search("How do I reset my password?")
print(result.passages)  # list[str] of top-k matching documents

# Use in a module
class QA(dspy.Module):
    def __init__(self, search):
        self.search = search
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.search(question).passages
        return self.answer(context=context, question=question)
```

### When to use Embeddings vs. ColBERTv2

| Scenario | Use |
|----------|-----|
| Quick prototyping with small-medium corpus | `dspy.retrievers.Embeddings` |
| Need a hosted, scalable retrieval server | `dspy.ColBERTv2` |
| Already have a vector store (Pinecone, Chroma, etc.) | Subclass `dspy.Retrieve` |
| Need full control over embeddings | `dspy.Embedder` + your own vector store |

## Building RAG pipelines

RAG is the most common use of retrieval in DSPy. The pattern: retrieve relevant passages, then generate an answer grounded in them.

### Basic RAG

```python
import dspy

class RAG(dspy.Module):
    def __init__(self, retriever, k=3):
        self.retrieve = retriever
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# With Embeddings retriever
embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=my_docs, k=5)
rag = RAG(retriever=search)
result = rag(question="How do refunds work?")
print(result.answer)
```

### RAG with source grounding

Use assertions to enforce that answers stay grounded in the retrieved context:

```python
class GroundedRAG(dspy.Module):
    def __init__(self, retriever):
        self.retrieve = retriever
        self.generate = dspy.ChainOfThought(
            "context, question -> answer, cited_sources: list[int]"
        )

    def forward(self, question):
        passages = self.retrieve(question).passages
        result = self.generate(context=passages, question=question)

        dspy.Suggest(
            len(result.cited_sources) > 0,
            "Answer should cite at least one source passage by index",
        )

        return dspy.Prediction(
            answer=result.answer,
            cited_sources=result.cited_sources,
            passages=passages,
        )
```

### Multi-hop RAG

When a question needs information from multiple documents, chain retrieval steps:

```python
class MultiHopRAG(dspy.Module):
    def __init__(self, retriever, hops=2):
        self.retrieve = retriever
        self.generate_query = [
            dspy.ChainOfThought("context, question -> search_query")
            for _ in range(hops)
        ]
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = []
        for hop in self.generate_query:
            query = hop(context=context, question=question).search_query
            new_passages = self.retrieve(query).passages
            context = list(dict.fromkeys(context + new_passages))  # deduplicate

        return self.answer(context=context, question=question)
```

## Configuring retrievers

There are two ways to wire up a retriever:

### Option 1: Global configuration with dspy.configure

```python
colbert = dspy.ColBERTv2(url="http://your-server:8893/api/search")
dspy.configure(lm=lm, rm=colbert)

# dspy.Retrieve() now uses colbert automatically
retriever = dspy.Retrieve(k=5)
```

### Option 2: Pass the retriever directly

```python
embedder = dspy.Embedder("openai/text-embedding-3-small")
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=docs, k=5)

class MyRAG(dspy.Module):
    def __init__(self):
        self.search = search  # use directly, no global config needed
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.search(question).passages
        return self.answer(context=context, question=question)
```

Option 2 is more explicit and avoids global state. Prefer it when your program uses a single retriever.

## The k parameter

The `k` parameter controls how many passages to retrieve. It can be set at multiple levels:

```python
# At init time
retriever = dspy.Retrieve(k=5)

# Override per call
result = retriever("query", k=10)

# In Embeddings constructor
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=docs, k=3)
```

**Choosing k:**
- Start with `k=3` to `k=5` for most tasks
- Increase `k` for questions that need broader context
- Decrease `k` for faster inference and lower token costs
- More passages means more context for the LM, but also more noise and higher cost
- Use evaluation to find the optimal `k` for your specific task

## Grounded generation with Citations

`dspy.experimental.Citations` is a module that makes the LM cite specific source passages in its output. Use it in RAG pipelines where you need verifiable references back to source documents.

```python
from dspy.experimental import Citations

# Citations wraps a generation step to produce cited output
cite = Citations(generate_query_or_answer="context, question -> answer")
result = cite(context=retrieved_passages, question=question)
# result.answer contains inline citations referencing specific passages
```

**When to use:** Any RAG pipeline where claims need to trace back to source documents — the module instructs the LM to ground its output in the provided passages and cite them.

**Note:** This is in `dspy.experimental` — the API may change in future releases. Check the [DSPy docs](https://dspy.ai/) for the latest usage.

For broader anti-hallucination patterns beyond citations, see `/ai-stopping-hallucinations`.

## Cross-references

- **Building custom modules** to wrap retrieval logic -- see `/dspy-modules`
- **Vector database setup** (Qdrant, Pinecone, ChromaDB, Weaviate) -- see `/dspy-qdrant`
- **End-to-end document search** with vector stores and chunking -- see `/ai-searching-docs`
- **Keeping answers grounded** and avoiding hallucination -- see `/ai-stopping-hallucinations`
- For worked examples, see [examples.md](examples.md)
- Not sure which skill to use next? Try `/ai-do` to get routed to the right one
