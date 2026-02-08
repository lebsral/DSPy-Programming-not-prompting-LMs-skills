# Multi-Agent Examples

## Example 1: Research team

A supervisor coordinates three specialists: a web researcher, an analyst, and a writer. The supervisor delegates tasks, collects results, and decides when the work is done.

### Agent modules

```python
import dspy
from langchain_community.tools import DuckDuckGoSearchRun

# Web researcher — uses search tools
search_tool = dspy.Tool.from_langchain(DuckDuckGoSearchRun())

class WebResearcher(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct(
            "topic, focus_area -> findings",
            tools=[search_tool],
            max_iters=5,
        )

    def forward(self, topic, focus_area=""):
        return self.agent(topic=topic, focus_area=focus_area or topic)


# Analyst — synthesizes research into insights
class AnalyzeFindings(dspy.Signature):
    """Analyze research findings and extract key insights, trends, and implications."""
    topic: str = dspy.InputField()
    raw_findings: str = dspy.InputField(desc="Research findings from various sources")
    insights: list[str] = dspy.OutputField(desc="Key insights and takeaways")
    trends: list[str] = dspy.OutputField(desc="Notable trends")
    recommendation: str = dspy.OutputField(desc="Overall recommendation based on analysis")

class Analyst(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought(AnalyzeFindings)

    def forward(self, topic, raw_findings):
        return self.analyze(topic=topic, raw_findings=raw_findings)


# Writer — produces the final report
class WriteReport(dspy.Signature):
    """Write a clear, well-structured report based on research and analysis."""
    topic: str = dspy.InputField()
    insights: str = dspy.InputField(desc="Analysis insights and recommendations")
    report: str = dspy.OutputField(desc="Well-structured report with sections and key takeaways")

class Writer(dspy.Module):
    def __init__(self):
        self.write = dspy.ChainOfThought(WriteReport)

    def forward(self, topic, insights):
        result = self.write(topic=topic, insights=insights)
        dspy.Suggest(
            len(result.report.split()) > 200,
            "Report should be substantive — at least 200 words",
        )
        return result
```

### Supervisor and graph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    topic: str
    messages: Annotated[list[dict], operator.add]
    results: dict
    current_agent: str
    status: str

class PlanNextStep(dspy.Signature):
    """Decide the next step in the research process."""
    topic: str = dspy.InputField()
    completed_steps: str = dspy.InputField()
    next_agent: str = dspy.OutputField(desc="One of: researcher, analyst, writer, done")
    instruction: str = dspy.OutputField(desc="What to tell the agent")

planner = dspy.ChainOfThought(PlanNextStep)
researcher = WebResearcher()
analyst = Analyst()
writer = Writer()

def supervisor_node(state: ResearchState) -> dict:
    completed = "\n".join(f"- {k}: done" for k in state["results"])
    result = planner(topic=state["topic"], completed_steps=completed or "Nothing yet")

    if result.next_agent == "done":
        return {"status": "done"}

    return {
        "current_agent": result.next_agent,
        "messages": [{"role": "supervisor", "content": result.instruction}],
    }

def researcher_node(state: ResearchState) -> dict:
    result = researcher(topic=state["topic"])
    return {
        "results": {**state["results"], "research": result.findings},
        "messages": [{"role": "researcher", "content": result.findings}],
    }

def analyst_node(state: ResearchState) -> dict:
    result = analyst(
        topic=state["topic"],
        raw_findings=state["results"].get("research", ""),
    )
    insights = f"Insights: {result.insights}\nTrends: {result.trends}\nRecommendation: {result.recommendation}"
    return {
        "results": {**state["results"], "analysis": insights},
        "messages": [{"role": "analyst", "content": insights}],
    }

def writer_node(state: ResearchState) -> dict:
    result = writer(
        topic=state["topic"],
        insights=state["results"].get("analysis", ""),
    )
    return {
        "results": {**state["results"], "report": result.report},
        "messages": [{"role": "writer", "content": result.report}],
    }

# Build graph
graph = StateGraph(ResearchState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("analyst", analyst_node)
graph.add_node("writer", writer_node)

graph.add_edge(START, "supervisor")

def route(state: ResearchState) -> str:
    if state["status"] == "done":
        return "done"
    return state["current_agent"]

graph.add_conditional_edges("supervisor", route, {
    "researcher": "researcher",
    "analyst": "analyst",
    "writer": "writer",
    "done": END,
})

graph.add_edge("researcher", "supervisor")
graph.add_edge("analyst", "supervisor")
graph.add_edge("writer", "supervisor")

app = graph.compile()
```

### Usage

```python
result = app.invoke({
    "topic": "Impact of AI on software development productivity in 2025",
    "messages": [],
    "results": {},
    "current_agent": "",
    "status": "in_progress",
})

print(result["results"]["report"])
# Full research report with findings, analysis, and recommendations
```

---

## Example 2: Support escalation (L1 → L2 specialists)

An L1 classifier routes tickets to specialized L2 agents. Each L2 agent has domain-specific tools and knowledge.

### L1 classifier

```python
from typing import Literal

class TriageTicket(dspy.Signature):
    """Classify the support ticket and decide which specialist team should handle it."""
    ticket: str = dspy.InputField()
    customer_tier: str = dspy.InputField(desc="free, pro, or enterprise")
    category: Literal["billing", "technical", "account", "security"] = dspy.OutputField()
    priority: Literal["low", "medium", "high", "critical"] = dspy.OutputField()

class L1Classifier(dspy.Module):
    def __init__(self):
        self.triage = dspy.ChainOfThought(TriageTicket)

    def forward(self, ticket, customer_tier):
        result = self.triage(ticket=ticket, customer_tier=customer_tier)
        # Enterprise customers with critical issues always escalate to human
        dspy.Assert(
            not (customer_tier == "enterprise" and result.priority == "critical"),
            "Enterprise critical issues go directly to human support",
        )
        return result
```

### L2 specialist agents

```python
class BillingResponse(dspy.Signature):
    """Handle a billing support issue. Be specific about amounts and dates."""
    ticket: str = dspy.InputField()
    account_info: str = dspy.InputField(desc="Customer's billing history")
    response: str = dspy.OutputField()
    action_needed: str = dspy.OutputField(desc="refund, credit, none, or escalate_to_human")

class TechnicalResponse(dspy.Signature):
    """Handle a technical support issue. Include specific troubleshooting steps."""
    ticket: str = dspy.InputField()
    docs: list[str] = dspy.InputField(desc="Relevant technical documentation")
    response: str = dspy.OutputField()
    resolved: bool = dspy.OutputField()

class BillingAgent(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought(BillingResponse)

    def forward(self, ticket, account_info):
        return self.respond(ticket=ticket, account_info=account_info)

class TechAgent(dspy.Module):
    def __init__(self, retriever):
        self.retriever = retriever
        self.respond = dspy.ChainOfThought(TechnicalResponse)

    def forward(self, ticket):
        docs = self.retriever(ticket).passages
        return self.respond(ticket=ticket, docs=docs)
```

### LangGraph escalation flow

```python
class SupportState(TypedDict):
    ticket: str
    customer_tier: str
    category: str
    priority: str
    response: str
    action_needed: str
    escalated_to_human: bool

classifier = L1Classifier()
billing_agent = BillingAgent()
tech_agent = TechAgent(retriever=tech_docs_retriever)

def classify_node(state: SupportState) -> dict:
    result = classifier(ticket=state["ticket"], customer_tier=state["customer_tier"])
    return {"category": result.category, "priority": result.priority}

def billing_node(state: SupportState) -> dict:
    account_info = lookup_billing(state["ticket"])  # your billing lookup
    result = billing_agent(ticket=state["ticket"], account_info=account_info)
    return {"response": result.response, "action_needed": result.action_needed}

def tech_node(state: SupportState) -> dict:
    result = tech_agent(ticket=state["ticket"])
    return {"response": result.response}

def human_escalation(state: SupportState) -> dict:
    return {"escalated_to_human": True, "response": "Escalated to human support team."}

def route_to_specialist(state: SupportState) -> str:
    if state["priority"] == "critical":
        return "human"
    return state["category"]

graph = StateGraph(SupportState)
graph.add_node("classify", classify_node)
graph.add_node("billing", billing_node)
graph.add_node("technical", tech_node)
graph.add_node("human", human_escalation)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route_to_specialist, {
    "billing": "billing",
    "technical": "technical",
    "account": "human",
    "security": "human",
    "human": "human",
})

# Billing agent can escalate to human for refunds
def billing_followup(state: SupportState) -> str:
    if state.get("action_needed") == "escalate_to_human":
        return "human"
    return "done"

graph.add_conditional_edges("billing", billing_followup, {"human": "human", "done": END})
graph.add_edge("technical", END)
graph.add_edge("human", END)

app = graph.compile()
```

### Usage

```python
# Technical issue — routed to tech agent
result = app.invoke({
    "ticket": "API returns 500 error when uploading files larger than 10MB",
    "customer_tier": "pro",
    "category": "", "priority": "", "response": "",
    "action_needed": "", "escalated_to_human": False,
})
print(result["response"])
# Specific troubleshooting steps from tech docs

# Critical billing issue — escalated to human
result = app.invoke({
    "ticket": "We were charged $50,000 instead of $500, need immediate resolution",
    "customer_tier": "enterprise",
    "category": "", "priority": "", "response": "",
    "action_needed": "", "escalated_to_human": False,
})
print(result["escalated_to_human"])  # True
```
