# Qdrant Examples

## Index and search with Qdrant + DSPy

```python
import dspy
from qdrant_client import QdrantClient, models
from dspy_qdrant import QdrantRM

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# --- Step 1: Index documents ---
client = QdrantClient("http://localhost:6333")
embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)

docs = [
    "Our refund policy allows returns within 30 days of purchase.",
    "To reset your password, visit Settings > Security > Change Password.",
    "Enterprise plans include SSO, audit logs, and a dedicated account manager.",
    "Free trials last 14 days. No credit card required.",
    "We support integration with Slack, Teams, and Discord.",
    "Data is encrypted at rest (AES-256) and in transit (TLS 1.3).",
    "API rate limits: 1000 requests/minute for Pro, 100 for Free.",
    "Billing is monthly. Switch to annual for a 20% discount.",
]

# Create collection
client.recreate_collection(
    collection_name="support_docs",
    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
)

# Embed and upsert
vectors = embedder(docs)
client.upsert(
    collection_name="support_docs",
    points=[
        models.PointStruct(id=i, vector=v, payload={"text": doc})
        for i, (doc, v) in enumerate(zip(docs, vectors))
    ],
)

# --- Step 2: Set up QdrantRM ---
retriever = QdrantRM(
    qdrant_collection_name="support_docs",
    qdrant_client_url="http://localhost:6333",
    k=3,
)

# --- Step 3: Build RAG pipeline ---
class SupportRAG(dspy.Module):
    def __init__(self):
        self.retrieve = retriever
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)

rag = SupportRAG()
result = rag(question="What's the refund policy?")
print(f"Answer: {result.answer}")
```

## Custom Pinecone retriever for DSPy

```python
from pinecone import Pinecone
import dspy

class PineconeRetriever(dspy.Retrieve):
    def __init__(self, index_name, embedder, namespace=None, k=3):
        super().__init__(k=k)
        pc = Pinecone()  # reads PINECONE_API_KEY from env
        self.index = pc.Index(index_name)
        self.embedder = embedder
        self.namespace = namespace

    def forward(self, query, k=None):
        k = k or self.k
        vector = self.embedder(query)
        results = self.index.query(
            vector=vector,
            top_k=k,
            include_metadata=True,
            namespace=self.namespace,
        )
        passages = [m["metadata"]["text"] for m in results["matches"]]
        return dspy.Prediction(passages=passages)

# Usage in a DSPy pipeline
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
retriever = PineconeRetriever("support-index", embedder, k=5)

class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = retriever
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)

rag = RAG()
print(rag(question="What integrations do you support?").answer)
```

## Custom ChromaDB retriever for prototyping

```python
import chromadb
import dspy

class ChromaRetriever(dspy.Retrieve):
    def __init__(self, collection_name, persist_dir="./chroma_db", k=3):
        super().__init__(k=k)
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(collection_name)

    def forward(self, query, k=None):
        k = k or self.k
        results = self.collection.query(query_texts=[query], n_results=k)
        return dspy.Prediction(passages=results["documents"][0])

# Quick prototype — index and search in one script
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("quick_test")
collection.add(
    documents=[
        "Python is a programming language.",
        "DSPy is a framework for programming LMs.",
        "Qdrant is a vector search engine.",
    ],
    ids=["1", "2", "3"],
)

retriever = ChromaRetriever("quick_test", k=2)
result = retriever("What is DSPy?")
print(result.passages)
# ['DSPy is a framework for programming LMs.', 'Python is a programming language.']
```
