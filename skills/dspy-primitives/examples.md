# DSPy Primitives — Examples

## 1. Image analysis with dspy.Image

Analyze product images to generate catalog descriptions automatically.

```python
import dspy
from typing import Literal

# Use a vision-capable model
lm = dspy.LM("openai/gpt-4o")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class CatalogEntry(dspy.Signature):
    """Generate a product catalog entry from the product image."""
    image: dspy.Image = dspy.InputField(desc="Product photo")
    category: str = dspy.InputField(desc="Product category for context")
    title: str = dspy.OutputField(desc="Short product title, max 10 words")
    description: str = dspy.OutputField(desc="Marketing description, 2-3 sentences")
    color: str = dspy.OutputField(desc="Primary color of the product")
    tags: list[str] = dspy.OutputField(desc="Search tags for the product")

cataloger = dspy.ChainOfThought(CatalogEntry)

# From a URL
result = cataloger(
    image=dspy.Image(url="https://example.com/products/red-sneaker.jpg"),
    category="footwear",
)
print(result.title)        # "Classic Red Canvas Sneaker"
print(result.description)  # "A bold red canvas sneaker with white rubber sole..."
print(result.color)        # "red"
print(result.tags)         # ["sneaker", "canvas", "red", "casual", "footwear"]

# From a local file
result = cataloger(
    image=dspy.Image(url="/photos/products/blue-jacket.png"),
    category="outerwear",
)
```

### Image comparison

```python
class QualityCheck(dspy.Signature):
    """Compare a reference product image to a manufacturing sample and flag defects."""
    reference: dspy.Image = dspy.InputField(desc="Reference product image")
    sample: dspy.Image = dspy.InputField(desc="Manufacturing sample photo")
    passes_qc: bool = dspy.OutputField(desc="Whether the sample matches the reference")
    defects: list[str] = dspy.OutputField(desc="List of defects found, empty if none")

checker = dspy.ChainOfThought(QualityCheck)
result = checker(
    reference=dspy.Image(url="/images/reference/widget-v2.png"),
    sample=dspy.Image(url="/images/samples/widget-batch-42-001.png"),
)
if not result.passes_qc:
    print(f"QC failed. Defects: {result.defects}")
```

## 2. Code review with dspy.Code

Analyze code for bugs, security issues, and suggest improvements.

```python
import dspy
from typing import Literal
from pydantic import BaseModel

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

class Issue(BaseModel):
    line_hint: str
    severity: str  # "info", "warning", "error"
    message: str
    suggestion: str

class CodeReview(dspy.Signature):
    """Review the code for bugs, security vulnerabilities, and style issues.
    Focus on actionable feedback a developer can fix immediately."""
    code: dspy.Code["python"] = dspy.InputField(desc="Code to review")
    context: str = dspy.InputField(desc="What this code is supposed to do")
    issues: list[Issue] = dspy.OutputField(desc="List of issues found")
    overall: Literal["approve", "request_changes"] = dspy.OutputField()

reviewer = dspy.ChainOfThought(CodeReview)

source_code = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result.fetchone()

def process_payment(amount, card_number):
    print(f"Processing payment of {amount} with card {card_number}")
    # TODO: implement actual payment
    return True
"""

result = reviewer(
    code=source_code,
    context="User lookup and payment processing for an e-commerce backend"
)

for issue in result.issues:
    print(f"[{issue.severity}] {issue.message}")
    print(f"  Suggestion: {issue.suggestion}")
    print()

print(f"Verdict: {result.overall}")
# [error] SQL injection vulnerability in get_user
#   Suggestion: Use parameterized queries: db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
#
# [error] Logging sensitive card number in process_payment
#   Suggestion: Mask the card number before logging: card_number[-4:]
# ...
# Verdict: request_changes
```

### Code generation and transformation

```python
class PortToTypeScript(dspy.Signature):
    """Port the Python code to idiomatic TypeScript.
    Preserve the logic and add proper type annotations."""
    python_code: dspy.Code["python"] = dspy.InputField(desc="Python source to port")
    typescript_code: dspy.Code["typescript"] = dspy.OutputField(desc="Equivalent TypeScript")
    notes: list[str] = dspy.OutputField(desc="Differences or caveats in the port")

porter = dspy.ChainOfThought(PortToTypeScript)
result = porter(python_code="def fibonacci(n: int) -> list[int]:\n    a, b = 0, 1\n    seq = []\n    for _ in range(n):\n        seq.append(a)\n        a, b = b, a + b\n    return seq")

print(result.typescript_code)
# function fibonacci(n: number): number[] {
#     let a = 0, b = 1;
#     const seq: number[] = [];
#     for (let i = 0; i < n; i++) {
#         seq.push(a);
#         [a, b] = [b, a + b];
#     }
#     return seq;
# }
print(result.notes)
# ["Python tuple unpacking replaced with destructuring assignment", ...]
```

## 3. Conversation with dspy.History

Build a multi-turn customer support chatbot that remembers context.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

class SupportChat(dspy.Signature):
    """You are a helpful customer support agent for an e-commerce store.
    Use the conversation history to maintain context across turns.
    Be concise and helpful."""
    history: dspy.History = dspy.InputField(desc="Prior conversation turns")
    question: str = dspy.InputField(desc="Customer's current message")
    answer: str = dspy.OutputField(desc="Support agent response")

class SupportBot(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought(SupportChat)
        self.turns = []

    def chat(self, message: str) -> str:
        history = dspy.History(messages=self.turns)
        result = self.respond(history=history, question=message)
        self.turns.append({"question": message, "answer": result.answer})
        return result.answer

    def reset(self):
        self.turns = []

# Usage
bot = SupportBot()

print(bot.chat("I ordered a blue jacket last week but haven't received it yet."))
# "I'm sorry to hear that. Could you provide your order number so I can look into it?"

print(bot.chat("It's ORDER-12345."))
# "Thanks! Let me check on ORDER-12345 for you..."

print(bot.chat("Can I change the shipping address?"))
# The bot remembers the order context from previous turns
```

### History with structured data

```python
from pydantic import BaseModel
from typing import Literal, Optional

class OrderLookup(dspy.Signature):
    """Look up order details and answer the customer's question.
    Use conversation history for context about which order is being discussed."""
    history: dspy.History = dspy.InputField(desc="Prior conversation")
    question: str = dspy.InputField(desc="Customer question")
    order_data: str = dspy.InputField(desc="Order data from the database, if available")
    answer: str = dspy.OutputField(desc="Helpful response")
    action: Optional[Literal["escalate", "refund", "reship", "none"]] = dspy.OutputField(
        desc="Action to take, if any"
    )

agent = dspy.ChainOfThought(OrderLookup)

# Simulate a conversation with database lookups
history = dspy.History(messages=[
    {"question": "Where is my order?", "answer": "Could you share your order number?"},
    {"question": "ORDER-789", "answer": "I found it. It shipped on March 10."},
])

result = agent(
    history=history,
    question="It's been a week and I still don't have it. This is unacceptable.",
    order_data="ORDER-789: shipped 2026-03-10, carrier: UPS, tracking: stuck at hub since 03-12"
)

print(result.answer)  # Empathetic response about the delay
print(result.action)  # "reship" or "escalate"
```
