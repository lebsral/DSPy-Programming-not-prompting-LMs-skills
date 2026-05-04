# Action-Taking AI Examples

## Calculator

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

def evaluate_math(expression: str) -> float:
    """Evaluate a mathematical expression and return the result."""
    return dspy.PythonInterpreter({}).execute(expression)

agent = dspy.ReAct("question -> answer: float", tools=[evaluate_math])

result = agent(question="What is (15 * 7 + 23) / 4?")
print(f"Answer: {result.answer}")  # 32.0

# Multi-step math
result = agent(question="If I have 3 boxes with 12 items each, and I remove 7 items total, how many are left?")
print(f"Answer: {result.answer}")  # 29.0
```

## Search + Calculator

```python
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual information."""
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return "\n".join([x["text"] for x in results])

def evaluate_math(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return dspy.PythonInterpreter({}).execute(expression)

agent = dspy.ReAct(
    "question -> answer",
    tools=[search_wikipedia, evaluate_math],
    max_iters=5,
)

# Requires both search and calculation
result = agent(
    question="What is the population of Tokyo divided by the population of Paris?"
)
print(result.answer)
```

## API-Calling AI

```python
import requests

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Replace with your actual weather API
    resp = requests.get(f"https://wttr.in/{city}?format=3")
    return resp.text

def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a ticker symbol."""
    # Replace with your actual stock API
    return f"{symbol}: $150.00"  # placeholder

agent = dspy.ReAct(
    "question -> answer",
    tools=[get_weather, get_stock_price],
    max_iters=3,
)

result = agent(question="What's the weather in San Francisco?")
print(result.answer)
```

## Research Bot with Custom Module

```python
def search(query: str) -> str:
    """Search for information on the web."""
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return "\n".join([x["text"] for x in results])

class ResearchBot(dspy.Module):
    """AI that researches a topic and provides a structured summary."""
    def __init__(self):
        self.researcher = dspy.ReAct(
            "topic -> findings",
            tools=[search],
            max_iters=5,
        )
        self.summarize = dspy.ChainOfThought(
            "topic, findings -> summary, key_facts: list[str]"
        )

    def forward(self, topic):
        research = self.researcher(topic=topic)
        return self.summarize(topic=topic, findings=research.findings)

bot = ResearchBot()
result = bot(topic="The history of the Python programming language")
print(f"Summary: {result.summary}")
print(f"Key facts: {result.key_facts}")
```

## Safety with Refine

```python
def safe_search(query: str) -> str:
    """Search for information from trusted sources."""
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return "\n".join([x["text"] for x in results])

class SafeBot(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct("question -> answer", tools=[safe_search], max_iters=3)

    def forward(self, question):
        return self.agent(question=question)


def safe_bot_reward(args, pred):
    """Reward function: hard require a non-empty answer, soft encourage confidence."""
    if not pred.answer or len(pred.answer.strip()) == 0:
        return 0.0  # hard: must produce an answer
    score = 1.0
    if "i don't know" in pred.answer.lower():
        score -= 0.2  # soft: encourage using the search tool more effectively
    return score

safe_bot = dspy.Refine(module=SafeBot(), N=3, reward_fn=safe_bot_reward, threshold=0.8)
```

## Optimizing Action-Taking AI

```python
# Training data
trainset = [
    dspy.Example(
        question="What is 9362158 divided by the year of birth of David Gregory?",
        answer="6780"
    ).with_inputs("question"),
    # ... more examples
]

# Optimize with MIPROv2 (good for reasoning instruction tuning)
optimizer = dspy.MIPROv2(metric=action_metric, auto="light")
optimized = optimizer.compile(agent, trainset=trainset)

# Save optimized version
optimized.save("optimized_agent.json")
```
