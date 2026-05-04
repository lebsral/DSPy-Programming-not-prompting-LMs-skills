# TwoStepAdapter Examples

## Example 1: Math problem solving with o3-mini

Using o3-mini for complex math with structured output:

```python
import dspy

main_lm = dspy.LM("openai/o3-mini")
extraction_lm = dspy.LM("openai/gpt-4o-mini")

adapter = dspy.TwoStepAdapter(
    main_lm=main_lm,
    extraction_lm=extraction_lm,
)

dspy.configure(lm=main_lm, adapter=adapter)


class MathSolver(dspy.Signature):
    """Solve the math problem step by step."""
    problem: str = dspy.InputField()
    solution: str = dspy.OutputField(desc="step-by-step solution")
    final_answer: str = dspy.OutputField(desc="numeric answer only")


solver = dspy.Predict(MathSolver)
result = solver(problem="A train travels 120km in 1.5 hours. It then speeds up by 20% for the next 2 hours. Total distance?")

print(f"Solution: {result.solution}")
print(f"Answer: {result.final_answer}")
```

## Example 2: Code review with reasoning model

Deep code analysis using extended reasoning:

```python
import dspy

# DeepSeek-R1 for deep reasoning about code
main_lm = dspy.LM("deepseek/deepseek-r1")
extraction_lm = dspy.LM("openai/gpt-4o-mini")

adapter = dspy.TwoStepAdapter(
    main_lm=main_lm,
    extraction_lm=extraction_lm,
)

dspy.configure(lm=main_lm, adapter=adapter)


class CodeReview(dspy.Signature):
    """Review the code for bugs, performance issues, and security vulnerabilities."""
    code: str = dspy.InputField()
    bugs: str = dspy.OutputField(desc="list of bugs found")
    performance: str = dspy.OutputField(desc="performance issues")
    security: str = dspy.OutputField(desc="security vulnerabilities")
    severity: str = dspy.OutputField(desc="overall severity - low, medium, high, critical")


reviewer = dspy.Predict(CodeReview)
result = reviewer(code="""
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
""")

print(f"Bugs: {result.bugs}")
print(f"Security: {result.security}")
print(f"Severity: {result.severity}")
```

## Example 3: Hybrid pipeline (fast + reasoning)

Using reasoning only where needed:

```python
import dspy

# Default fast model
fast_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=fast_lm)

# Reasoning setup for hard problems
reasoning_lm = dspy.LM("openai/o3-mini")
extraction_lm = dspy.LM("openai/gpt-4o-mini")
reasoning_adapter = dspy.TwoStepAdapter(
    main_lm=reasoning_lm,
    extraction_lm=extraction_lm,
)


class SmartPipeline(dspy.Module):
    def __init__(self):
        # Fast: classify difficulty
        self.classify = dspy.Predict("question -> difficulty: str")

        # Fast: answer easy questions
        self.easy_answer = dspy.ChainOfThought("question -> answer")

        # Reasoning: answer hard questions
        self.hard_answer = dspy.ChainOfThought("question -> answer")
        self.hard_answer.lm = reasoning_lm
        self.hard_answer.adapter = reasoning_adapter

    def forward(self, question):
        difficulty = self.classify(question=question).difficulty

        if "hard" in difficulty.lower() or "complex" in difficulty.lower():
            return self.hard_answer(question=question)
        else:
            return self.easy_answer(question=question)


pipeline = SmartPipeline()
result = pipeline(question="Prove that there are infinitely many primes")
print(result.answer)
```
