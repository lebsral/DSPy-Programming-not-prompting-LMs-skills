# dspy.Refine Examples

## Example 1: Iterative text improvement with quality criteria

A content writing pipeline that refines blog post introductions until they meet multiple quality criteria: appropriate length, includes a hook, and avoids filler phrases.

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

# Define the writing module
class WriteIntro(dspy.Module):
    def __init__(self):
        self.write = dspy.ChainOfThought(
            "topic, audience -> introduction"
        )

    def forward(self, topic, audience):
        return self.write(topic=topic, audience=audience)


# Define quality criteria as a graduated reward function
FILLER_PHRASES = [
    "in today's world",
    "it's no secret that",
    "in this article",
    "have you ever wondered",
    "let's dive in",
]

def intro_quality(args, pred):
    """Score introduction quality on multiple criteria (0.0 to 1.0)."""
    intro = pred.introduction
    score = 0.0

    # Criterion 1: Length between 50-150 words (0.3 points)
    word_count = len(intro.split())
    if 50 <= word_count <= 150:
        score += 0.3
    elif 30 <= word_count <= 200:
        score += 0.15  # partial credit

    # Criterion 2: No filler phrases (0.3 points)
    has_filler = any(phrase in intro.lower() for phrase in FILLER_PHRASES)
    if not has_filler:
        score += 0.3

    # Criterion 3: Mentions the target audience (0.2 points)
    audience = args["audience"].lower()
    if audience in intro.lower() or any(
        word in intro.lower() for word in audience.split()
    ):
        score += 0.2

    # Criterion 4: Has at least 2 sentences (0.2 points)
    sentence_count = intro.count('.') + intro.count('!') + intro.count('?')
    if sentence_count >= 2:
        score += 0.2

    return score


# Wrap with Refine -- up to 4 attempts, accept at 0.8+
refined_writer = dspy.Refine(
    module=WriteIntro(),
    N=4,
    reward_fn=intro_quality,
    threshold=0.8,
)

# Use it
result = refined_writer(
    topic="Using type hints in Python",
    audience="backend developers",
)
print(result.introduction)
print(f"Score: {intro_quality({'topic': 'Using type hints in Python', 'audience': 'backend developers'}, result)}")
```

What this demonstrates:

- **Graduated reward function** with four weighted criteria -- Refine picks the best attempt even if none score perfectly
- **Wrapping a custom module** -- `WriteIntro` is a standard `dspy.Module` with `forward()`
- **Multiple input fields** -- both `topic` and `audience` are available in the `args` dict
- **Practical quality checks** -- length bounds, filler phrase detection, audience relevance, sentence structure

## Example 2: Self-correcting code generation

A code generation pipeline that writes a Python function and validates it by parsing and running basic checks. Refine retries with feedback when the generated code has syntax errors or fails validation.

```python
import ast
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # or "anthropic/claude-sonnet-4-5-20250929", etc.

# Signature with structured output expectations
class GenerateFunction(dspy.Signature):
    """Generate a Python function that solves the given task."""
    task_description: str = dspy.InputField(desc="What the function should do")
    function_name: str = dspy.InputField(desc="Name for the generated function")
    code: str = dspy.OutputField(desc="Complete Python function code, no markdown fences")
    explanation: str = dspy.OutputField(desc="Brief explanation of the approach")


class CodeGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateFunction)

    def forward(self, task_description, function_name):
        return self.generate(
            task_description=task_description,
            function_name=function_name,
        )


def valid_python_function(args, pred):
    """Score generated code on syntax, structure, and basic quality."""
    code = pred.code
    expected_name = args["function_name"]
    score = 0.0

    # Strip markdown fences if present
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    # Criterion 1: Valid Python syntax (0.4 points)
    try:
        tree = ast.parse(code)
        score += 0.4
    except SyntaxError:
        return 0.0  # no point checking further

    # Criterion 2: Contains a function definition (0.2 points)
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]
    if functions:
        score += 0.2

    # Criterion 3: Function has the expected name (0.2 points)
    func_names = [f.name for f in functions]
    if expected_name in func_names:
        score += 0.2

    # Criterion 4: Has a docstring (0.1 points)
    for func in functions:
        if (
            func.body
            and isinstance(func.body[0], ast.Expr)
            and isinstance(func.body[0].value, ast.Constant)
            and isinstance(func.body[0].value.value, str)
        ):
            score += 0.1
            break

    # Criterion 5: Has a return statement (0.1 points)
    for func in functions:
        for node in ast.walk(func):
            if isinstance(node, ast.Return):
                score += 0.1
                break
        break  # only check first matching function

    return score


# Wrap with Refine -- up to 5 attempts for code generation
refined_coder = dspy.Refine(
    module=CodeGenerator(),
    N=5,
    reward_fn=valid_python_function,
    threshold=0.9,
)

# Generate a function
result = refined_coder(
    task_description="Calculate the nth Fibonacci number using memoization. Handle negative inputs by raising ValueError.",
    function_name="fibonacci",
)

print("Generated code:")
print(result.code)
print(f"\nExplanation: {result.explanation}")

# Verify the score
score = valid_python_function(
    {"task_description": "...", "function_name": "fibonacci"},
    result,
)
print(f"Quality score: {score}")
```

What this demonstrates:

- **AST-based validation** -- uses Python's `ast` module for reliable syntax and structure checking, no regex heuristics
- **Class-based signature** -- `GenerateFunction` with typed fields and descriptions gives the LM clear expectations
- **Early exit in reward** -- returns 0.0 immediately on syntax error since no other criteria matter
- **Higher N (5)** -- code generation benefits from more attempts because valid code is harder to produce
- **Practical code checks** -- correct function name, docstring presence, return statement -- criteria you would check in a real code review
- **Feedback loop** -- when attempt 1 has a syntax error, Refine tells the LM what went wrong so attempt 2 can fix it
