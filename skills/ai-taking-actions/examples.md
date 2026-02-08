# Action-Taking AI Examples

## Calculator

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
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

## Safety Assertions

```python
def safe_search(query: str) -> str:
    """Search for information from trusted sources."""
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return "\n".join([x["text"] for x in results])

class SafeBot(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct("question -> answer", tools=[safe_search], max_iters=3)

    def forward(self, question):
        result = self.agent(question=question)
        dspy.Assert(
            len(result.answer) > 0,
            "Must provide an answer"
        )
        dspy.Suggest(
            "I don't know" not in result.answer.lower(),
            "Try harder to find the answer using the search tool"
        )
        return result
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
