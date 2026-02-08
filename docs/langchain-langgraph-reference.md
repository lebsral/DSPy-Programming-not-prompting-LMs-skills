# LangChain & LangGraph Quick Reference

> Condensed API reference for using LangChain and LangGraph alongside DSPy. For full docs, see [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/).

## Setup

```bash
pip install langchain langchain-community langgraph
# Plus provider-specific packages:
pip install langchain-openai langchain-anthropic langchain-chroma
```

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
```

## Document Loaders

Load data from 100+ sources into a standard `Document(page_content, metadata)` format:

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
    NotionDBLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
    DirectoryLoader,
)

# PDF
docs = PyPDFLoader("report.pdf").load()

# Web page
docs = WebBaseLoader("https://example.com/docs").load()

# Directory of files
docs = DirectoryLoader("./data/", glob="**/*.txt", loader_cls=TextLoader).load()

# CSV
docs = CSVLoader("data.csv", source_column="url").load()

# JSON
docs = JSONLoader("data.json", jq_schema=".records[]", content_key="text").load()
```

Each document has:
- `page_content` — the text
- `metadata` — source info (file path, page number, etc.)

## Text Splitters

Break documents into chunks for embedding and retrieval:

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)

# General-purpose (recommended default)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Markdown-aware (splits by headings)
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)
chunks = splitter.split_text(markdown_text)

# Token-based (for strict token budgets)
splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
```

## Vector Stores

Store and search embeddings:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create from documents
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# Search
results = vectorstore.similarity_search("How do refunds work?", k=5)

# As a retriever (for use in chains)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

Other vector stores follow the same pattern: `Pinecone.from_documents(...)`, `FAISS.from_documents(...)`, `Weaviate.from_documents(...)`.

## LangChain Tools

Define tools for agents:

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    # your search logic
    return results

@tool
def query_database(sql: str) -> str:
    """Run a SQL query against the analytics database."""
    # your database logic
    return results
```

### Convert LangChain tools to DSPy

```python
import dspy

# One-liner: convert any LangChain tool for use in DSPy agents
dspy_tool = dspy.Tool.from_langchain(search_web)

# Use in a DSPy ReAct agent
agent = dspy.ReAct("question -> answer", tools=[dspy_tool])
```

### Pre-built tool packages

```bash
pip install langchain-community  # Wikipedia, DuckDuckGo, requests, etc.
```

```python
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Convert to DSPy
dspy_search = dspy.Tool.from_langchain(search)
dspy_wiki = dspy.Tool.from_langchain(wikipedia)
```

## LangGraph StateGraph

Build stateful, multi-step workflows with conditional routing and cycles:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

# 1. Define state
class MyState(TypedDict):
    messages: Annotated[list, operator.add]  # append-only list
    step: str

# 2. Define nodes (functions that take and return state)
def process(state: MyState) -> dict:
    return {"messages": ["processed"], "step": "done"}

def review(state: MyState) -> dict:
    return {"messages": ["reviewed"]}

# 3. Build graph
graph = StateGraph(MyState)
graph.add_node("process", process)
graph.add_node("review", review)
graph.add_edge(START, "process")
graph.add_edge("process", "review")
graph.add_edge("review", END)

# 4. Compile and run
app = graph.compile()
result = app.invoke({"messages": [], "step": ""})
```

### Conditional edges

```python
def should_review(state: MyState) -> str:
    if state["step"] == "needs_review":
        return "review"
    return "done"

graph.add_conditional_edges("process", should_review, {"review": "review", "done": END})
```

## LangGraph Checkpointing

Persist state across sessions (conversation memory, resumable workflows):

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Each thread_id is a separate session
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": ["hello"]}, config=config)

# Resume later — state is preserved
result = app.invoke({"messages": ["follow-up"]}, config=config)
```

For production, use persistent checkpointers:

```python
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver(conn_string="postgresql://...")
```

## LangGraph Human-in-the-Loop

Pause execution for human review before sensitive actions:

```python
# Pause BEFORE a node runs
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_action"],
)

# Run until interrupt
result = app.invoke(input, config)
# -> pauses before "execute_action"

# Human reviews, then resumes
result = app.invoke(None, config)  # continues from checkpoint
```

## LangGraph Multi-Agent Patterns

### Supervisor pattern

A supervisor agent routes tasks to specialist agents:

```python
from langgraph.graph import StateGraph, START, END

def supervisor(state):
    # Decide which agent to call next
    if state["task_type"] == "research":
        return {"next_agent": "researcher"}
    return {"next_agent": "writer"}

def route(state) -> str:
    return state["next_agent"]

graph = StateGraph(State)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route, {"researcher": "researcher", "writer": "writer"})
graph.add_edge("researcher", "supervisor")  # report back
graph.add_edge("writer", END)
```

### Parallel execution with Send

Fan out to multiple agents simultaneously:

```python
from langgraph.constants import Send

def fan_out(state):
    # Send tasks to multiple agents in parallel
    return [Send("worker", {"task": t}) for t in state["tasks"]]

graph.add_conditional_edges("splitter", fan_out)
```

## DSPy Module as LangGraph Node

The key integration pattern — wrap DSPy modules as LangGraph node functions:

```python
import dspy
from langgraph.graph import StateGraph

# DSPy module
class AnswerQuestion(dspy.Signature):
    """Answer the question based on context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

answer_module = dspy.ChainOfThought(AnswerQuestion)

# LangGraph node that calls the DSPy module
def answer_node(state):
    result = answer_module(context=state["context"], question=state["question"])
    return {"answer": result.answer}

# Wire into graph
graph = StateGraph(MyState)
graph.add_node("answer", answer_node)
```

This gives you LangGraph's state management + routing with DSPy's optimizable prompts.

## Links

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
