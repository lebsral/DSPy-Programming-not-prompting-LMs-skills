# dspy-react -- Worked Examples

## Example 1: Search agent with a single tool

A simple agent that answers questions by searching a knowledge base. Demonstrates the basic ReAct pattern with one tool.

```python
import dspy


# --- Tool ---

KNOWLEDGE_BASE = {
    "dspy": "DSPy is a framework for programming language models with optimizable modules.",
    "react": "ReAct is an agent pattern that combines reasoning with tool use in a loop.",
    "signatures": "DSPy signatures declare input/output behavior as typed specs like 'question -> answer'.",
    "optimizers": "DSPy optimizers tune prompts or weights to improve program accuracy.",
}


def search_docs(query: str) -> str:
    """Search the documentation knowledge base for information about a topic."""
    query_lower = query.lower()
    results = []
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower or any(word in value.lower() for word in query_lower.split()):
            results.append(value)
    if results:
        return " ".join(results)
    return "No results found for that query."


# --- Agent ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

agent = dspy.ReAct("question -> answer", tools=[search_docs], max_iters=3)

# Test it
result = agent(question="What is DSPy and how does it use signatures?")
print(result.answer)

# Inspect the reasoning trace
dspy.inspect_history(n=3)
```

Key points:
- One tool is enough for many use cases -- the agent decides when to call it
- `max_iters=3` keeps the agent focused since a single search is usually sufficient
- The tool returns plain strings; the agent interprets them and formulates an answer
- `inspect_history` shows the full Thought-Action-Observation trace for debugging


## Example 2: Multi-tool customer support agent

A support agent with tools to look up users, check orders, and search a FAQ. Demonstrates routing between multiple tools and wrapping ReAct in a custom module.

```python
import dspy
import json


# --- Tools ---

USERS_DB = {
    "alice@example.com": {"name": "Alice", "plan": "pro", "joined": "2024-01-15"},
    "bob@example.com": {"name": "Bob", "plan": "free", "joined": "2024-06-01"},
}

ORDERS_DB = {
    "ORD-001": {"user": "alice@example.com", "status": "shipped", "eta": "March 20"},
    "ORD-002": {"user": "bob@example.com", "status": "processing", "eta": "March 25"},
}

FAQ = {
    "refund": "Refunds are processed within 5-7 business days after approval.",
    "upgrade": "You can upgrade your plan at any time from Settings > Billing.",
    "cancel": "To cancel, go to Settings > Billing > Cancel Subscription.",
}


def lookup_user(email: str) -> str:
    """Look up a user account by their email address. Returns account details."""
    user = USERS_DB.get(email)
    if user:
        return json.dumps(user)
    return f"No user found with email {email}."


def check_order(order_id: str) -> str:
    """Check the status of an order by its order ID (e.g., ORD-001)."""
    order = ORDERS_DB.get(order_id.upper())
    if order:
        return json.dumps(order)
    return f"No order found with ID {order_id}."


def search_faq(topic: str) -> str:
    """Search the FAQ for help articles about a topic."""
    topic_lower = topic.lower()
    matches = []
    for key, value in FAQ.items():
        if key in topic_lower or topic_lower in key:
            matches.append(f"{key}: {value}")
    if matches:
        return "\n".join(matches)
    return "No FAQ articles found for that topic."


# --- Agent module ---

class SupportAgent(dspy.Module):
    """Customer support agent that looks up accounts, orders, and FAQ articles."""

    def __init__(self):
        self.agent = dspy.ReAct(
            "question, context -> answer",
            tools=[lookup_user, check_order, search_faq],
            max_iters=5,
        )

    def forward(self, question: str):
        context = (
            "You are a helpful customer support agent. "
            "Use lookup_user for account questions (requires email), "
            "check_order for order status (requires order ID like ORD-001), "
            "and search_faq for general help topics. "
            "Be friendly and specific in your answers."
        )
        result = self.agent(question=question, context=context)

        # Ensure the response is helpful
        dspy.Suggest(
            len(result.answer.strip()) > 30,
            "Provide a detailed, helpful response with specific information",
        )

        return result


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

support = SupportAgent()

# Test with different question types
questions = [
    "What plan is alice@example.com on?",
    "Where is my order ORD-002?",
    "How do I get a refund?",
    "Can you check if bob@example.com has any orders?",  # requires chaining tools
]

for q in questions:
    result = support(question=q)
    print(f"\nQ: {q}")
    print(f"A: {result.answer}")


# --- Optimization ---

def support_metric(example, prediction, trace=None):
    """Check if the answer contains the expected key information."""
    answer = prediction.answer.lower()
    # Check that expected keywords appear in the answer
    return all(kw.lower() in answer for kw in example.expected_keywords)


trainset = [
    dspy.Example(
        question="What plan is alice@example.com on?",
        expected_keywords=["pro"],
    ).with_inputs("question"),
    dspy.Example(
        question="Where is my order ORD-001?",
        expected_keywords=["shipped", "march 20"],
    ).with_inputs("question"),
    dspy.Example(
        question="How do I cancel my subscription?",
        expected_keywords=["settings", "billing"],
    ).with_inputs("question"),
]

# optimizer = dspy.BootstrapFewShot(metric=support_metric, max_bootstrapped_demos=3)
# optimized_support = optimizer.compile(support, trainset=trainset)
# optimized_support.save("support_agent.json")
```

Key points:
- Three tools covering different domains -- the agent picks the right one based on the question
- The agent can chain tools: look up a user, then check their orders
- Context string guides the agent on when to use each tool
- `dspy.Suggest` enforces minimum response quality
- The metric checks for expected keywords rather than exact match, which works well for open-ended agent responses


## Example 3: Data lookup agent with API calls

An agent that fetches data from REST APIs to answer questions. Demonstrates real HTTP calls, error handling in tools, and structured output.

```python
import dspy
import requests
from typing import Literal


# --- Tools ---

def get_github_repo(repo: str) -> str:
    """Get information about a GitHub repository. Pass the full name like 'stanfordnlp/dspy'."""
    try:
        response = requests.get(
            f"https://api.github.com/repos/{repo}",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        if response.status_code == 404:
            return f"Repository '{repo}' not found."
        response.raise_for_status()
        data = response.json()
        return (
            f"Name: {data['full_name']}\n"
            f"Description: {data['description']}\n"
            f"Stars: {data['stargazers_count']}\n"
            f"Language: {data['language']}\n"
            f"Open issues: {data['open_issues_count']}\n"
            f"Last updated: {data['updated_at']}"
        )
    except requests.Timeout:
        return "Error: GitHub API request timed out. Try again."
    except requests.RequestException as e:
        return f"Error fetching repository info: {str(e)}"


def get_github_issues(repo: str, state: str = "open") -> str:
    """Get recent issues for a GitHub repository. Pass repo as 'owner/name' and state as 'open' or 'closed'."""
    try:
        response = requests.get(
            f"https://api.github.com/repos/{repo}/issues",
            params={"state": state, "per_page": 5, "sort": "updated"},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        response.raise_for_status()
        issues = response.json()
        if not issues:
            return f"No {state} issues found for {repo}."
        lines = []
        for issue in issues:
            lines.append(f"#{issue['number']}: {issue['title']} ({issue['state']})")
        return "\n".join(lines)
    except requests.Timeout:
        return "Error: GitHub API request timed out. Try again."
    except requests.RequestException as e:
        return f"Error fetching issues: {str(e)}"


def search_pypi(package_name: str) -> str:
    """Search for a Python package on PyPI and return its details."""
    try:
        response = requests.get(
            f"https://pypi.org/pypi/{package_name}/json",
            timeout=10,
        )
        if response.status_code == 404:
            return f"Package '{package_name}' not found on PyPI."
        response.raise_for_status()
        data = response.json()
        info = data["info"]
        return (
            f"Name: {info['name']}\n"
            f"Version: {info['version']}\n"
            f"Summary: {info['summary']}\n"
            f"Author: {info['author']}\n"
            f"License: {info['license']}\n"
            f"Home page: {info['home_page'] or info.get('project_url', 'N/A')}"
        )
    except requests.Timeout:
        return "Error: PyPI request timed out. Try again."
    except requests.RequestException as e:
        return f"Error searching PyPI: {str(e)}"


# --- Signature ---

class ResearchAnswer(dspy.Signature):
    """Research a question about open-source software using available tools."""
    question: str = dspy.InputField(desc="A question about open-source packages or repositories")
    answer: str = dspy.OutputField(desc="A detailed answer with specific data from the tools")
    confidence: Literal["high", "medium", "low"] = dspy.OutputField(
        desc="Confidence based on whether the tools returned useful data"
    )


# --- Agent module ---

class OSSResearcher(dspy.Module):
    """Agent that researches open-source software using GitHub and PyPI APIs."""

    def __init__(self):
        self.agent = dspy.ReAct(
            ResearchAnswer,
            tools=[get_github_repo, get_github_issues, search_pypi],
            max_iters=6,
        )

    def forward(self, question: str):
        result = self.agent(question=question)

        # Ensure we got real data, not just a guess
        dspy.Suggest(
            result.confidence != "low",
            "If confidence is low, try calling another tool for more information",
        )

        return result


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

researcher = OSSResearcher()

questions = [
    "How many stars does stanfordnlp/dspy have and what is the latest PyPI version?",
    "What are the most recent open issues in the stanfordnlp/dspy repo?",
    "Compare the dspy and langchain packages on PyPI.",
]

for q in questions:
    print(f"\nQ: {q}")
    result = researcher(question=q)
    print(f"A: {result.answer}")
    print(f"Confidence: {result.confidence}")


# --- Optimization ---

def research_metric(example, prediction, trace=None):
    """Score based on confidence and whether key facts are in the answer."""
    has_data = prediction.confidence in ("high", "medium")
    has_answer = len(prediction.answer.strip()) > 50
    return has_data + 0.5 * has_answer

# trainset = [
#     dspy.Example(question="How many stars does stanfordnlp/dspy have?").with_inputs("question"),
#     dspy.Example(question="What version is dspy on PyPI?").with_inputs("question"),
# ]
# optimizer = dspy.MIPROv2(metric=research_metric, auto="light")
# optimized = optimizer.compile(researcher, trainset=trainset)
# optimized.save("oss_researcher.json")
```

Key points:
- **Real API calls** with proper error handling -- every tool catches timeouts and HTTP errors, returning a useful message instead of crashing
- **Class-based signature** with typed output (`confidence: Literal[...]`) gives structured results
- The agent chains tools naturally: fetch repo info from GitHub, then check PyPI for the package version
- `dspy.Suggest` nudges the agent to gather more data when confidence is low
- `MIPROv2` is a good optimizer choice for agents because it tunes the reasoning instructions
