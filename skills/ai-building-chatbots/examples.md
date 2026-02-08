# Chatbot Examples

## Example 1: Customer support bot

A support bot that classifies intent, retrieves help articles, responds, and escalates unresolved issues.

### Signatures

```python
import dspy
from typing import Literal

class ClassifyIntent(dspy.Signature):
    """Classify the customer's intent."""
    conversation_history: str = dspy.InputField()
    user_message: str = dspy.InputField()
    intent: Literal["billing", "technical", "account", "general", "escalate"] = dspy.OutputField()

class SupportResponse(dspy.Signature):
    """Generate a helpful support response grounded in the help docs."""
    conversation_history: str = dspy.InputField()
    docs: list[str] = dspy.InputField(desc="Relevant help articles")
    user_message: str = dspy.InputField()
    intent: str = dspy.InputField()
    response: str = dspy.OutputField(desc="Helpful, empathetic support response")
    resolved: bool = dspy.OutputField(desc="Whether the issue appears resolved")
```

### DSPy module

```python
class SupportBot(dspy.Module):
    def __init__(self, retriever):
        self.classify = dspy.Predict(ClassifyIntent)
        self.respond = dspy.ChainOfThought(SupportResponse)
        self.retriever = retriever

    def forward(self, conversation_history, user_message):
        intent = self.classify(
            conversation_history=conversation_history,
            user_message=user_message,
        ).intent

        docs = self.retriever(f"{intent} {user_message}").passages

        result = self.respond(
            conversation_history=conversation_history,
            docs=docs,
            user_message=user_message,
            intent=intent,
        )

        dspy.Suggest(
            len(result.response.split()) < 150,
            "Keep support responses concise",
        )

        return dspy.Prediction(
            response=result.response,
            intent=intent,
            resolved=result.resolved,
        )
```

### LangGraph conversation flow

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

class SupportState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    intent: str
    resolved: bool
    escalated: bool
    turn_count: int

bot = SupportBot(retriever=my_retriever)

def handle_message(state: SupportState) -> dict:
    history = "\n".join(f"{m['role']}: {m['content']}" for m in state["messages"][:-1][-10:])
    user_msg = state["messages"][-1]["content"]

    result = bot(conversation_history=history, user_message=user_msg)

    return {
        "messages": [{"role": "assistant", "content": result.response}],
        "intent": result.intent,
        "resolved": result.resolved,
        "turn_count": state["turn_count"] + 1,
    }

def should_escalate(state: SupportState) -> str:
    if state["intent"] == "escalate":
        return "escalate"
    if state["turn_count"] > 5 and not state["resolved"]:
        return "escalate"
    return "done"

def escalate_node(state: SupportState) -> dict:
    return {
        "messages": [{"role": "assistant", "content": "Let me connect you with a specialist who can help further."}],
        "escalated": True,
    }

graph = StateGraph(SupportState)
graph.add_node("handle", handle_message)
graph.add_node("escalate", escalate_node)
graph.add_edge(START, "handle")
graph.add_conditional_edges("handle", should_escalate, {"escalate": "escalate", "done": END})
graph.add_edge("escalate", END)

app = graph.compile(checkpointer=MemorySaver())
```

### Usage

```python
config = {"configurable": {"thread_id": "session-001"}}

# Turn 1
result = app.invoke(
    {"messages": [{"role": "user", "content": "I was charged twice for my subscription"}],
     "intent": "", "resolved": False, "escalated": False, "turn_count": 0},
    config=config,
)
print(result["messages"][-1]["content"])
# "I'm sorry to hear about the double charge. Let me look into your billing..."

# Turn 2
result = app.invoke(
    {"messages": [{"role": "user", "content": "It happened on January 15th, order #12345"}]},
    config=config,
)
print(result["messages"][-1]["content"])
# "I can see the duplicate charge for order #12345. I'll process a refund..."
```

---

## Example 2: FAQ assistant with memory

A simple FAQ bot that remembers what was already asked and avoids repeating itself.

### DSPy module

```python
class FAQResponse(dspy.Signature):
    """Answer the user's question from the FAQ. If they already asked something similar, reference the earlier answer instead of repeating."""
    previous_topics: list[str] = dspy.InputField(desc="Topics already covered in this session")
    faq_entries: list[str] = dspy.InputField(desc="Relevant FAQ entries")
    user_message: str = dspy.InputField()
    response: str = dspy.OutputField()
    topic: str = dspy.OutputField(desc="Topic of this question for tracking")

class FAQBot(dspy.Module):
    def __init__(self, retriever):
        self.respond = dspy.ChainOfThought(FAQResponse)
        self.retriever = retriever

    def forward(self, previous_topics, user_message):
        faqs = self.retriever(user_message).passages
        return self.respond(
            previous_topics=previous_topics,
            faq_entries=faqs,
            user_message=user_message,
        )
```

### LangGraph with topic tracking

```python
class FAQState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    topics_covered: list[str]

faq_bot = FAQBot(retriever=my_retriever)

def answer(state: FAQState) -> dict:
    user_msg = state["messages"][-1]["content"]
    result = faq_bot(
        previous_topics=state["topics_covered"],
        user_message=user_msg,
    )
    return {
        "messages": [{"role": "assistant", "content": result.response}],
        "topics_covered": state["topics_covered"] + [result.topic],
    }

graph = StateGraph(FAQState)
graph.add_node("answer", answer)
graph.add_edge(START, "answer")
graph.add_edge("answer", END)

app = graph.compile(checkpointer=MemorySaver())
```

### Usage

```python
config = {"configurable": {"thread_id": "faq-session-1"}}

result = app.invoke(
    {"messages": [{"role": "user", "content": "What's your refund policy?"}],
     "topics_covered": []},
    config=config,
)
# Answers from FAQ docs

result = app.invoke(
    {"messages": [{"role": "user", "content": "How long do refunds take?"}]},
    config=config,
)
# References the earlier refund answer, adds processing time details

result = app.invoke(
    {"messages": [{"role": "user", "content": "Tell me about refunds again"}]},
    config=config,
)
# "As I mentioned earlier, [summary]. Is there something specific I missed?"
```
