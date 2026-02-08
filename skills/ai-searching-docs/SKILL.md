---
name: ai-searching-docs
description: Build AI that searches your documents and answers questions. Use when building a knowledge base, help center Q&A, chatting with documents, answering questions from a database, search-and-answer over internal docs, customer support bot, or FAQ system. Powered by DSPy RAG (retrieval-augmented generation).
---

# Build AI-Powered Document Search

Guide the user through building an AI that searches documents and answers questions accurately. Uses DSPy's RAG (retrieval-augmented generation) pattern — retrieve relevant passages, then generate an answer grounded in them.

## Step 1: Understand the setup

Ask the user:
1. **What documents are you searching?** (PDFs, web pages, database, help articles, etc.)
2. **What kind of questions will users ask?** (factual lookups, how-to questions, multi-step research?)
3. **Do you have a search backend already?** (Elasticsearch, Pinecone, ChromaDB, pgvector, etc.)
4. **Do questions need info from multiple documents?** (simple lookup vs. combining info)

## Step 2: Build the search-and-answer pipeline

### Basic: search then answer

```python
import dspy

class AnswerFromDocs(dspy.Signature):
    """Answer the question based on the given context."""
    context: list[str] = dspy.InputField(desc="Relevant passages from the knowledge base")
    question: str = dspy.InputField(desc="User's question")
    answer: str = dspy.OutputField(desc="Answer grounded in the context")

class DocSearch(dspy.Module):
    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.answer = dspy.ChainOfThought(AnswerFromDocs)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)
```

### Configure the search backend

DSPy supports multiple search backends. Set up via `dspy.configure`:

```python
# ColBERTv2 (hosted)
colbert = dspy.ColBERTv2(url="http://your-server:port/endpoint")
dspy.configure(lm=lm, rm=colbert)

# Or wrap your own search (Elasticsearch, Pinecone, pgvector, etc.)
class MySearchBackend(dspy.Retrieve):
    def forward(self, query, k=None):
        k = k or self.k
        # Your search logic here
        results = your_search_function(query, top_k=k)
        return dspy.Prediction(passages=[r["text"] for r in results])
```

## Step 3: Set up a vector store

If you don't have a search backend yet, set one up. ChromaDB is the simplest option for getting started:

### ChromaDB setup

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_docs")
```

### Load and chunk documents

Split documents into passages before adding them to the vector store. Sentence-based chunking works well for most use cases:

```python
import re

def chunk_text(text, max_sentences=5):
    """Split text into chunks of N sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks

# Load and chunk your documents
for doc in documents:
    chunks = chunk_text(doc["text"])
    collection.add(
        documents=chunks,
        ids=[f"{doc['id']}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": doc["source"]}] * len(chunks),
    )
```

### Custom embeddings

ChromaDB uses its default embedding function, but you can swap in others:

```python
# SentenceTransformers (local, free)
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# OpenAI embeddings (API, paid)
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
ef = OpenAIEmbeddingFunction(api_key="...", model_name="text-embedding-3-small")

collection = client.get_or_create_collection("my_docs", embedding_function=ef)
```

### Chunking strategies

| Strategy | How it works | Best for |
|----------|-------------|----------|
| Sentence-based | Split on sentence boundaries | Articles, docs, help pages |
| Fixed-size | Split every N characters with overlap | Long unstructured text |
| Paragraph | Split on double newlines | Well-structured documents |
| Overlap | Fixed-size with N-character overlap between chunks | When context at chunk boundaries matters |

### Wire it up as a DSPy retriever

```python
class ChromaRetriever(dspy.Retrieve):
    def __init__(self, collection, k=3):
        super().__init__(k=k)
        self.collection = collection

    def forward(self, query, k=None):
        k = k or self.k
        results = self.collection.query(query_texts=[query], n_results=k)
        return dspy.Prediction(passages=results["documents"][0])

# Use it
retriever = ChromaRetriever(collection)
dspy.configure(lm=lm, rm=retriever)
```

## Step 4: Multi-document search (for complex questions)

When questions need info from multiple places:

```python
class GenerateSearchQuery(dspy.Signature):
    """Generate a search query to find missing information."""
    context: list[str] = dspy.InputField(desc="Information gathered so far")
    question: str = dspy.InputField(desc="The question to answer")
    query: str = dspy.OutputField(desc="Search query to find missing information")

class MultiStepSearch(dspy.Module):
    def __init__(self, num_passages=3, num_searches=2):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(num_searches)]
        self.answer = dspy.ChainOfThought(AnswerFromDocs)

    def forward(self, question):
        context = []

        for hop in self.generate_query:
            query = hop(context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        return self.answer(context=context, question=question)

def deduplicate(passages):
    seen = set()
    result = []
    for p in passages:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result
```

## Step 5: Test the quality

```python
def search_metric(example, prediction, trace=None):
    # Exact match (simple)
    return prediction.answer == example.answer

# Or use an AI judge for open-ended answers
class JudgeAnswer(dspy.Signature):
    """Is the predicted answer correct given the expected answer?"""
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    is_correct: bool = dspy.OutputField()

def judge_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeAnswer)
    result = judge(
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=prediction.answer,
    )
    return result.is_correct
```

## Step 6: Improve accuracy

```python
optimizer = dspy.BootstrapFewShot(metric=search_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(DocSearch(), trainset=trainset)
```

## Key patterns

- **Always use `ChainOfThought`** for the answer step — reasoning helps ground answers in the documents
- **Include context in the signature** so the AI knows to use the retrieved passages
- **Multi-step search for complex questions** — if one search isn't enough, chain search queries
- **Use `dspy.Assert`** to ensure answers actually cite the documents
- **Separate search from answer generation** — optimize each independently

## Additional resources

- For worked examples, see [examples.md](examples.md)
- Need to summarize docs instead of answering questions? Use `/ai-summarizing`
- Use `/ai-serving-apis` to put your document search behind a REST API
- Next: `/ai-improving-accuracy` to measure and improve your AI
