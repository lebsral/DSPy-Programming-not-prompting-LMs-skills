# AI Reasoning — Worked Examples

## Example 1: Complex customer question solver

Handle nuanced customer questions that need multi-step reasoning to answer correctly.

### Setup

```python
import dspy
from pydantic import BaseModel, Field

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
```

### Signatures and module

```python
class AnalyzeQuestion(dspy.Signature):
    """Break down a complex customer question into sub-questions."""
    question: str = dspy.InputField(desc="Customer's question")
    context: str = dspy.InputField(desc="Relevant product/policy information")
    sub_questions: list[str] = dspy.OutputField(desc="Sub-questions to answer first")

class AnswerSubQuestion(dspy.Signature):
    """Answer one specific sub-question using the context."""
    sub_question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()

class SynthesizeAnswer(dspy.Signature):
    """Combine sub-answers into a complete, helpful response."""
    question: str = dspy.InputField(desc="Original customer question")
    sub_answers: list[str] = dspy.InputField(desc="Answers to each sub-question")
    response: str = dspy.OutputField(desc="Complete answer addressing all parts")

class ComplexQuestionSolver(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought(AnalyzeQuestion)
        self.answer_sub = dspy.ChainOfThought(AnswerSubQuestion)
        self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)

    def forward(self, question, context):
        # Break down the question
        analysis = self.analyze(question=question, context=context)

        # Answer each sub-question
        sub_answers = []
        for sq in analysis.sub_questions:
            result = self.answer_sub(sub_question=sq, context=context)
            sub_answers.append(f"{sq}: {result.answer}")

        # Synthesize into a complete response
        final = self.synthesize(question=question, sub_answers=sub_answers)

        return dspy.Prediction(
            sub_questions=analysis.sub_questions,
            sub_answers=sub_answers,
            response=final.response,
        )
```

### Usage

```python
solver = ComplexQuestionSolver()

result = solver(
    question="If I upgrade from the Pro plan to Enterprise mid-billing-cycle, do I get a prorated refund for Pro, and does the Enterprise trial period still apply?",
    context="""
    Pricing policy:
    - Pro plan: $49/month, billed monthly
    - Enterprise plan: $199/month, billed annually
    - Mid-cycle upgrades: prorated credit applied to new plan
    - Enterprise trial: 14-day free trial for new customers only
    - Existing customers upgrading: no trial, immediate billing
    """,
)

print(result.sub_questions)
# ["Does a mid-cycle upgrade get prorated?", "Does the Enterprise trial apply to upgrades?", "How is the billing difference calculated?"]
print(result.response)
# Clear answer covering proration, no trial for upgrades, and billing details
```

### Metric

```python
class JudgeCustomerResponse(dspy.Signature):
    """Judge if the response correctly and completely answers the customer question."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    response: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    is_correct: bool = dspy.OutputField(desc="Factually correct given the context")
    is_complete: bool = dspy.OutputField(desc="Addresses all parts of the question")
    is_clear: bool = dspy.OutputField(desc="Easy for a customer to understand")

def customer_qa_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeCustomerResponse)
    result = judge(
        question=example.question,
        context=example.context,
        response=prediction.response,
        gold_answer=example.response,
    )
    return (result.is_correct + result.is_complete + result.is_clear) / 3
```

---

## Example 2: Multi-step data analysis

Analyze data by breaking the problem into computation steps.

### Signatures and module

```python
class PlanAnalysis(dspy.Signature):
    """Plan the steps needed to analyze this data question."""
    question: str = dspy.InputField(desc="The analysis question")
    data_description: str = dspy.InputField(desc="Description of available data")
    steps: list[str] = dspy.OutputField(desc="Ordered computation steps")

class ComputeStep(dspy.Signature):
    """Perform one computation step of the analysis."""
    step_description: str = dspy.InputField()
    data_description: str = dspy.InputField()
    prior_results: list[str] = dspy.InputField(desc="Results from previous steps")
    result: str = dspy.OutputField(desc="The computed result for this step")

class DataAnalyzer(dspy.Module):
    def __init__(self):
        self.plan = dspy.ChainOfThought(PlanAnalysis)
        self.compute = dspy.ProgramOfThought(ComputeStep)

    def forward(self, question, data_description):
        # Plan the analysis
        analysis_plan = self.plan(
            question=question,
            data_description=data_description,
        )

        # Execute each step
        prior_results = []
        for step in analysis_plan.steps:
            result = self.compute(
                step_description=step,
                data_description=data_description,
                prior_results=prior_results,
            )
            prior_results.append(f"{step}: {result.result}")

        return dspy.Prediction(
            plan=analysis_plan.steps,
            step_results=prior_results,
            answer=prior_results[-1] if prior_results else "No result",
        )
```

### Usage

```python
analyzer = DataAnalyzer()

result = analyzer(
    question="Which product category had the highest month-over-month growth in Q4?",
    data_description="""
    Monthly revenue by category:
    Electronics: Oct=$120k, Nov=$135k, Dec=$180k
    Clothing: Oct=$80k, Nov=$95k, Dec=$110k
    Home: Oct=$60k, Nov=$55k, Dec=$70k
    """,
)

print(result.plan)
# ["Calculate MoM growth rates for each category", "Compare growth rates", "Identify the highest"]
print(result.answer)
# "Electronics had the highest MoM growth at 33.3% (Nov→Dec)"
```

---

## Example 3: Planning and scheduling assistant

Use Self-Discovery reasoning to plan complex tasks with constraints.

### Signatures and module

```python
PLANNING_STRATEGIES = [
    "Identify all constraints and hard deadlines",
    "Find dependencies — what must happen before what",
    "Estimate effort for each task",
    "Look for tasks that can happen in parallel",
    "Identify the critical path (longest chain of dependencies)",
    "Build in buffer time for unknowns",
    "Consider resource availability",
]

class SelectPlanningStrategies(dspy.Signature):
    """Select the most relevant planning strategies for this scenario."""
    scenario: str = dspy.InputField()
    strategies: list[str] = dspy.InputField()
    selected: list[str] = dspy.OutputField(desc="2-4 most relevant strategies")

class TaskItem(BaseModel):
    name: str
    duration: str = Field(description="Estimated duration, e.g. '2 days'")
    depends_on: list[str] = Field(description="Names of tasks that must complete first")
    assigned_to: str = Field(description="Who should do this, or 'unassigned'")

class CreatePlan(dspy.Signature):
    """Create a project plan based on the reasoning strategies."""
    scenario: str = dspy.InputField()
    strategies: list[str] = dspy.InputField(desc="Planning strategies to apply")
    tasks: list[TaskItem] = dspy.OutputField(desc="Ordered list of tasks")
    critical_path: list[str] = dspy.OutputField(desc="Tasks on the critical path")
    estimated_total: str = dspy.OutputField(desc="Total estimated time")
    risks: list[str] = dspy.OutputField(desc="Key risks to watch")

class PlanningAssistant(dspy.Module):
    def __init__(self):
        self.select = dspy.ChainOfThought(SelectPlanningStrategies)
        self.plan = dspy.ChainOfThought(CreatePlan)

    def forward(self, scenario):
        # Select relevant strategies
        selected = self.select(
            scenario=scenario,
            strategies=PLANNING_STRATEGIES,
        ).selected

        # Create the plan
        result = self.plan(
            scenario=scenario,
            strategies=selected,
        )

        dspy.Suggest(
            len(result.tasks) >= 3,
            "A real plan should have at least 3 tasks"
        )
        dspy.Suggest(
            len(result.risks) >= 1,
            "Every plan has at least one risk — identify it"
        )

        return result
```

### Usage

```python
planner = PlanningAssistant()

result = planner(scenario="""
We need to migrate our database from PostgreSQL to a new managed service.
Constraints: zero downtime for the API, 3 engineers available, must complete
in 2 weeks, 50GB of data, the API handles 1000 req/sec during peak hours.
""")

for task in result.tasks:
    deps = f" (after: {', '.join(task.depends_on)})" if task.depends_on else ""
    print(f"  [{task.duration}] {task.name}{deps} — {task.assigned_to}")

print(f"\nCritical path: {' → '.join(result.critical_path)}")
print(f"Total estimate: {result.estimated_total}")
print(f"Risks: {result.risks}")
```

### Metric

```python
class JudgePlan(dspy.Signature):
    """Judge whether the plan is realistic and complete."""
    scenario: str = dspy.InputField()
    tasks: list[str] = dspy.InputField(desc="Task names from the plan")
    risks: list[str] = dspy.InputField()
    addresses_constraints: bool = dspy.OutputField(desc="Plan accounts for all stated constraints")
    dependencies_make_sense: bool = dspy.OutputField(desc="Task ordering is logical")
    is_actionable: bool = dspy.OutputField(desc="Someone could follow this plan and execute it")

def planning_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgePlan)
    result = judge(
        scenario=example.scenario,
        tasks=[t.name for t in prediction.tasks],
        risks=prediction.risks,
    )
    return (result.addresses_constraints + result.dependencies_make_sense + result.is_actionable) / 3

optimizer = dspy.BootstrapFewShot(metric=planning_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(PlanningAssistant(), trainset=trainset)
```
