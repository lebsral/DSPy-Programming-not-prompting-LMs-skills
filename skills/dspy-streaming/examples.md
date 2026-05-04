# Streaming Examples

## Example 1: Chat interface with typing indicator

A chat endpoint that streams tokens and shows "AI is thinking..." status:

```python
import dspy
from dspy.streaming import streamify, StreamListener, StatusMessage

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

chatbot = dspy.ChainOfThought("message, history -> response")

response_listener = StreamListener(signature_field_name="response")

streaming_chat = streamify(
    chatbot,
    stream_listeners=[response_listener],
)


async def handle_message(message: str, history: str):
    """Stream response tokens to the frontend."""
    tokens = []
    async for chunk in streaming_chat(message=message, history=history):
        if isinstance(chunk, StatusMessage):
            yield {"type": "status", "text": chunk.message}
        elif hasattr(chunk, "response"):
            tokens.append(chunk.response)
            yield {"type": "token", "text": chunk.response}
        elif isinstance(chunk, dspy.Prediction):
            yield {"type": "done", "full_response": chunk.response}
```

## Example 2: Streaming ReAct agent with progress updates

An agent that searches and summarizes, streaming each step:

```python
import dspy
from dspy.streaming import streamify, StreamListener, StatusMessage

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


def search_web(query: str) -> str:
    """Search the web for information."""
    # Your search implementation
    return "Search results for: " + query


def get_page(url: str) -> str:
    """Fetch the content of a web page."""
    # Your fetch implementation
    return "Page content..."


agent = dspy.ReAct("question -> answer", tools=[search_web, get_page])

answer_listener = StreamListener(
    signature_field_name="answer",
    allow_reuse=True,  # Critical for agents
)

streaming_agent = streamify(
    agent,
    stream_listeners=[answer_listener],
)

# Consume with progress display
async for chunk in streaming_agent(question="What are the latest DSPy features?"):
    if isinstance(chunk, StatusMessage):
        print(f"\n[{chunk.message}]")
    elif hasattr(chunk, "answer"):
        print(chunk.answer, end="", flush=True)
    elif isinstance(chunk, dspy.Prediction):
        print(f"\n\nFinal: {chunk.answer}")
```

## Example 3: WebSocket streaming with multiple fields

Streaming a multi-field module over WebSocket:

```python
import dspy
from dspy.streaming import streamify, StreamListener

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class AnalyzeReview(dspy.Signature):
    """Analyze a product review."""
    review: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")
    summary: str = dspy.OutputField(desc="one-sentence summary")
    suggestions: str = dspy.OutputField(desc="improvement suggestions for the product team")


analyzer = dspy.Predict(AnalyzeReview)

# Listen to the long-form field that benefits from streaming
summary_listener = StreamListener(signature_field_name="summary")
suggestions_listener = StreamListener(signature_field_name="suggestions")

streaming_analyzer = streamify(
    analyzer,
    stream_listeners=[summary_listener, suggestions_listener],
)


async def websocket_handler(ws, review_text: str):
    """Stream analysis results over WebSocket."""
    async for chunk in streaming_analyzer(review=review_text):
        if hasattr(chunk, "summary"):
            await ws.send_json({"field": "summary", "token": chunk.summary})
        elif hasattr(chunk, "suggestions"):
            await ws.send_json({"field": "suggestions", "token": chunk.suggestions})
        elif isinstance(chunk, dspy.Prediction):
            await ws.send_json({
                "field": "complete",
                "sentiment": chunk.sentiment,
                "summary": chunk.summary,
                "suggestions": chunk.suggestions,
            })
```
