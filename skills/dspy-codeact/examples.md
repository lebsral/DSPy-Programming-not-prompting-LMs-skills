# CodeAct Examples

## Data Analysis Agent

An agent that answers questions about data by writing computation code:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Simulate a data source -- in production, this could query a database or API
SALES_DATA = {
    "Q1": {"revenue": 150000, "units": 1200, "returns": 45},
    "Q2": {"revenue": 210000, "units": 1800, "returns": 62},
    "Q3": {"revenue": 185000, "units": 1500, "returns": 38},
    "Q4": {"revenue": 290000, "units": 2400, "returns": 71},
}

def get_quarterly_data(quarter: str) -> str:
    """Get sales data for a specific quarter (Q1, Q2, Q3, or Q4).

    Returns revenue, units sold, and number of returns.
    """
    if quarter not in SALES_DATA:
        return f"No data for {quarter}. Valid quarters: Q1, Q2, Q3, Q4"
    data = SALES_DATA[quarter]
    return f"Revenue: ${data['revenue']}, Units: {data['units']}, Returns: {data['returns']}"

def get_all_quarters() -> str:
    """List all available quarters."""
    return ", ".join(SALES_DATA.keys())

agent = dspy.CodeAct(
    "question -> answer",
    tools=[get_quarterly_data, get_all_quarters],
    max_iters=8,
)

# The agent writes code to fetch each quarter, compute averages, find trends
result = agent(
    question="What was the average revenue per unit across all quarters, "
    "and which quarter had the best ratio?"
)
print(result.answer)

# Another query -- the agent figures out the computation approach
result = agent(
    question="What is the total return rate (returns/units) for the year, "
    "and how does Q3 compare to the yearly average?"
)
print(result.answer)
```

Why CodeAct fits here: the agent needs to fetch data from multiple quarters, do arithmetic across them, and compare ratios. Writing code to loop, divide, and compare is more natural than making individual tool calls.

## File Processing Agent

An agent that processes structured text content:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Simulated file store -- in production, read from disk or object storage
FILE_STORE = {
    "employees.csv": "name,department,salary\nAlice,Engineering,120000\nBob,Marketing,95000\nCarol,Engineering,135000\nDave,Marketing,88000\nEve,Engineering,110000",
    "config.txt": "max_retries=3\ntimeout=30\nlog_level=INFO\napi_url=https://api.example.com",
    "report.txt": "Monthly Report\n\nTotal sales: 45000\nNew customers: 120\nChurn rate: 2.3%\nNPS score: 72",
}

def read_file(filename: str) -> str:
    """Read the contents of a file. Returns the full text content."""
    if filename not in FILE_STORE:
        available = ", ".join(FILE_STORE.keys())
        return f"File not found: {filename}. Available files: {available}"
    return FILE_STORE[filename]

def list_files() -> str:
    """List all available files."""
    return "\n".join(f"- {name}" for name in FILE_STORE.keys())

def write_result(filename: str, content: str) -> str:
    """Write processed content to a result file."""
    FILE_STORE[filename] = content
    return f"Wrote {len(content)} characters to {filename}"

agent = dspy.CodeAct(
    "task -> result",
    tools=[read_file, list_files, write_result],
    max_iters=8,
)

# The agent reads the CSV, parses it in code, and computes the answer
result = agent(
    task="Read employees.csv and calculate the average salary per department. "
    "Write the results to summary.txt."
)
print(result.result)
print("Generated file:", FILE_STORE.get("summary.txt", "not created"))
```

Why CodeAct fits here: parsing CSV data, splitting strings, aggregating by group, and formatting output are all natural code operations. ReAct would struggle to do this with just tool calls.

## Math and Computation Agent

An agent that solves math problems by writing code to work through them:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def gcd(a: int, b: int) -> int:
    """Calculate the greatest common divisor of two numbers."""
    while b:
        a, b = b, a % b
    return a

def factorial(n: int) -> int:
    """Calculate the factorial of n."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

agent = dspy.CodeAct(
    "problem -> answer",
    tools=[is_prime, gcd, factorial],
    max_iters=6,
)

# The agent writes code to iterate and test primes
result = agent(
    problem="Find the sum of all prime numbers between 1 and 50."
)
print(f"Sum of primes 1-50: {result.answer}")

# Multi-step computation
result = agent(
    problem="What is the GCD of factorial(8) and factorial(6)? "
    "Express the answer as a product of prime factors."
)
print(f"Answer: {result.answer}")

# Complex logic the agent figures out on its own
result = agent(
    problem="Find the smallest number greater than 100 that is prime "
    "and whose digits sum to a prime number."
)
print(f"Answer: {result.answer}")
```

Why CodeAct fits here: math problems often require writing loops, conditionals, and combining multiple operations. The agent can write a search loop to find numbers meeting complex criteria -- something that would be awkward with individual tool calls.

## Wrapping CodeAct in a Module with Safety Checks

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

def compute(expression: str) -> str:
    """Evaluate a mathematical expression and return the result as a string."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

class MathAgent(dspy.Module):
    def __init__(self):
        self.agent = dspy.CodeAct(
            "problem -> answer: str",
            tools=[compute],
            max_iters=5,
        )

    def forward(self, problem):
        result = self.agent(problem=problem)
        dspy.Assert(
            result.answer.strip() != "",
            "Agent must produce a non-empty answer",
        )
        dspy.Suggest(
            "error" not in result.answer.lower(),
            "Answer should not contain errors -- try a different approach",
        )
        return result

agent = MathAgent()
result = agent(problem="What is 2^10 + 3^7?")
print(result.answer)
```

## Optimizing a CodeAct Agent

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

agent = dspy.CodeAct("problem -> answer", tools=[is_prime], max_iters=5)

# Training data
trainset = [
    dspy.Example(
        problem="How many prime numbers are between 10 and 30?",
        answer="6",
    ).with_inputs("problem"),
    dspy.Example(
        problem="What is the sum of the first 5 prime numbers?",
        answer="28",
    ).with_inputs("problem"),
    dspy.Example(
        problem="Is 97 prime? Answer yes or no.",
        answer="yes",
    ).with_inputs("problem"),
    # Add more examples for better optimization...
]

def metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# Optimize
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
optimized = optimizer.compile(agent, trainset=trainset)

# Evaluate
evaluator = Evaluate(devset=trainset, metric=metric, num_threads=2)
score = evaluator(optimized)
print(f"Score: {score}")

# Save for production
optimized.save("math_codeact_agent.json")
```
