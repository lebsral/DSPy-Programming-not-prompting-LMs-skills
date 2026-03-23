---
name: dspy-primitives
description: "Use when working with non-text inputs like images, audio, or code — DSPy's typed wrappers for multimodal data in signatures. Common scenarios: processing images alongside text, handling audio transcription inputs, working with code files as typed inputs, building multimodal AI pipelines, or declaring non-string field types in signatures for structured data. Related: ai-parsing-data, dspy-signatures. Also: "dspy.Image", "dspy.History", "multimodal DSPy", "image input in DSPy signature", "process images with DSPy", "audio input in DSPy", "typed fields in signatures", "non-text data in DSPy", "vision model with DSPy", "GPT-4V with DSPy", "Claude vision with DSPy", "multimodal pipeline", "image classification with DSPy", "pass images to language model in DSPy", "conversation history type", "handle binary data in DSPy", "structured types beyond strings"."
---

# DSPy Primitives

Guide the user through DSPy's built-in primitive types for multimodal inputs, code handling, and conversation history.

## What are primitives

Primitives are DSPy's custom types that go beyond plain strings. They let you pass images, audio, code, and conversation history directly into signatures. DSPy handles the formatting, encoding, and adapter logic so the LM receives the data in the right format for its provider.

The four primitives:

| Primitive | Purpose | Typical use case |
|-----------|---------|------------------|
| `dspy.Image` | Images from URLs, files, or bytes | Vision tasks, image analysis, multimodal Q&A |
| `dspy.Audio` | Audio from files, URLs, or arrays | Transcription, audio classification |
| `dspy.Code` | Code with language annotation | Code generation, code review, analysis |
| `dspy.History` | Conversation turns | Chatbots, multi-turn dialogue, follow-up questions |

## dspy.Image

Wraps an image from any source into a format the LM can process. DSPy normalizes the input into a base64 data URI or plain URL automatically.

### Constructor

```python
dspy.Image(url=<source>, download=False, verify=True)
```

**Parameters:**

- **`url`** — the image source. Accepts:
  - `str` — HTTP/HTTPS URL, GS URL, or local file path
  - `bytes` — raw image bytes
  - `PIL.Image.Image` — a PIL image instance
  - `dict` — `{"url": value}` (legacy form)
  - An already-encoded data URI
- **`download`** (`bool`, default `False`) — whether to download remote URLs to infer MIME type
- **`verify`** (`bool`, default `True`) — whether to verify SSL certificates. Set `False` for self-signed certs.

### Usage in signatures

```python
import dspy

lm = dspy.LM("openai/gpt-4o")  # or any vision-capable LM
dspy.configure(lm=lm)

class DescribeImage(dspy.Signature):
    """Describe what you see in the image."""
    image: dspy.Image = dspy.InputField(desc="Image to analyze")
    description: str = dspy.OutputField(desc="Detailed description of the image")

describer = dspy.Predict(DescribeImage)

# From a URL
result = describer(image=dspy.Image(url="https://example.com/photo.jpg"))
print(result.description)

# From a local file
result = describer(image=dspy.Image(url="/path/to/photo.png"))

# From PIL
from PIL import Image as PILImage
pil_img = PILImage.open("photo.png")
result = describer(image=dspy.Image(url=pil_img))
```

### Multiple images

```python
class CompareImages(dspy.Signature):
    """Compare two images and describe the differences."""
    image_a: dspy.Image = dspy.InputField(desc="First image")
    image_b: dspy.Image = dspy.InputField(desc="Second image")
    differences: str = dspy.OutputField(desc="Key differences between the images")
```

## dspy.Audio

Wraps audio data for LMs that support native audio input. Audio is encoded as base64 internally.

### Creating Audio objects

```python
# From a local file
audio = dspy.Audio.from_file("recording.wav")

# From a URL
audio = dspy.Audio.from_url("https://example.com/clip.mp3")

# From a numpy array (e.g., from a microphone or audio processing)
import numpy as np
audio = dspy.Audio.from_array(samples, sampling_rate=16000, format="wav")

# Direct instantiation with base64 data
audio = dspy.Audio(data="<base64-string>", audio_format="wav")
```

### Usage in signatures

```python
import dspy

lm = dspy.LM("openai/gpt-4o-audio-preview")  # or any audio-capable LM
dspy.configure(lm=lm)

class TranscribeAudio(dspy.Signature):
    """Transcribe the spoken content in the audio."""
    audio: dspy.Audio = dspy.InputField(desc="Audio recording to transcribe")
    transcript: str = dspy.OutputField(desc="Transcribed text")

transcriber = dspy.Predict(TranscribeAudio)
result = transcriber(audio=dspy.Audio.from_file("meeting.wav"))
print(result.transcript)
```

### Audio classification

```python
from typing import Literal

class ClassifyAudio(dspy.Signature):
    """Classify the type of audio content."""
    audio: dspy.Audio = dspy.InputField(desc="Audio clip to classify")
    category: Literal["speech", "music", "ambient", "silence"] = dspy.OutputField()
    language: str = dspy.OutputField(desc="Detected language if speech, else 'N/A'")
```

## dspy.Code

Wraps code with a language annotation. DSPy formats it as a markdown code block so the LM sees properly delimited, syntax-aware code.

### Language specification

Use bracket notation to specify the language:

```python
dspy.Code["python"]   # Python code
dspy.Code["java"]     # Java code
dspy.Code["sql"]      # SQL code
dspy.Code["rust"]     # Rust code
# ... any language string works
```

The language tag tells DSPy to format the code as a fenced markdown block (` ```python ... ``` `) and guides the LM on syntax expectations.

### Usage in signatures

**Code generation (output):**

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

class GenerateCode(dspy.Signature):
    """Generate Python code that solves the given problem."""
    problem: str = dspy.InputField(desc="Problem description")
    code: dspy.Code["python"] = dspy.OutputField(desc="Working Python solution")

generator = dspy.Predict(GenerateCode)
result = generator(problem="Write a function that checks if a string is a palindrome")
print(result.code)
```

**Code analysis (input):**

```python
class ReviewCode(dspy.Signature):
    """Review the code for bugs, performance issues, and style problems."""
    code: dspy.Code["python"] = dspy.InputField(desc="Code to review")
    issues: list[str] = dspy.OutputField(desc="List of issues found")
    severity: Literal["clean", "minor", "major", "critical"] = dspy.OutputField()

reviewer = dspy.ChainOfThought(ReviewCode)
result = reviewer(code="def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)")
print(result.issues)    # ["No memoization — exponential time complexity", ...]
print(result.severity)  # "major"
```

**Code transformation (input and output):**

```python
class ConvertCode(dspy.Signature):
    """Convert the Python code to equivalent Java code."""
    python_code: dspy.Code["python"] = dspy.InputField(desc="Python source code")
    java_code: dspy.Code["java"] = dspy.OutputField(desc="Equivalent Java code")
```

## dspy.History

Represents conversation history as a list of message turns. Use it to build multi-turn chatbots and follow-up interactions in DSPy.

### Creating History objects

```python
# From prior conversation turns
history = dspy.History(messages=[
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris"},
])

# Using field names that match your signature
history = dspy.History(messages=[
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Germany?", "answer": "Berlin"},
])
```

History objects are **immutable** (frozen). Create a new History to add turns.

### Usage in signatures

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

class Chat(dspy.Signature):
    """Answer the user's question given the conversation history."""
    history: dspy.History = dspy.InputField(desc="Prior conversation turns")
    question: str = dspy.InputField(desc="Current user question")
    answer: str = dspy.OutputField(desc="Response to the user")

chatbot = dspy.Predict(Chat)
```

### Building conversation incrementally

Capture each response and append it to history for the next turn:

```python
chatbot = dspy.Predict(Chat)

# Turn 1
result = chatbot(
    history=dspy.History(messages=[]),
    question="What is the capital of France?"
)
print(result.answer)  # Paris

# Turn 2 — include previous turn in history
history = dspy.History(messages=[
    {"question": "What is the capital of France?", "answer": result.answer}
])
result = chatbot(
    history=history,
    question="What is its population?"
)
print(result.answer)  # About 2.1 million in the city proper...

# Turn 3 — append again
history = dspy.History(messages=[
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is its population?", "answer": result.answer},
])
result = chatbot(
    history=history,
    question="How does that compare to London?"
)
```

### Helper pattern for managing history

```python
class Chatbot(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought(Chat)
        self.turns = []

    def forward(self, question):
        history = dspy.History(messages=self.turns)
        result = self.respond(history=history, question=question)
        self.turns.append({"question": question, "answer": result.answer})
        return result
```

## Combining primitives in signatures

You can mix primitives with regular typed fields in the same signature:

```python
class AnalyzeScreenshot(dspy.Signature):
    """Analyze a UI screenshot and generate test code for the visible elements."""
    screenshot: dspy.Image = dspy.InputField(desc="Screenshot of the UI")
    framework: str = dspy.InputField(desc="Test framework to use, e.g. 'playwright'")
    test_code: dspy.Code["python"] = dspy.OutputField(desc="Generated test code")
    element_count: int = dspy.OutputField(desc="Number of interactive elements found")
```

```python
class AudioChat(dspy.Signature):
    """Respond to a user's audio message in a conversation."""
    history: dspy.History = dspy.InputField(desc="Prior conversation turns")
    audio_message: dspy.Audio = dspy.InputField(desc="User's spoken message")
    response: str = dspy.OutputField(desc="Text response to the user")
```

## Provider requirements

Not all LM providers support all primitives natively:

| Primitive | Requires |
|-----------|----------|
| `dspy.Image` | A vision-capable model (GPT-4o, Claude 3+, Gemini, etc.) |
| `dspy.Audio` | An audio-capable model (GPT-4o-audio-preview, Gemini, etc.) |
| `dspy.Code` | Any LM (formatted as markdown code blocks) |
| `dspy.History` | Any LM (formatted as conversation turns) |

DSPy's adapter system handles the provider-specific formatting. You write the signature once; DSPy translates it for the target LM.

## Cross-references

- **Defining signatures** — see `/dspy-signatures`
- **Using signatures with modules** — see `/dspy-modules`, `/dspy-predict`, `/dspy-chain-of-thought`
- **Building chatbots with History** — see `/ai-building-chatbots`
- **Full working examples** — see [examples.md](examples.md)
