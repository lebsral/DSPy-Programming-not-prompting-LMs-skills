# dspy-best-of-n -- Worked Examples

## Example 1: Best-of-3 code generation with test-based selection

Generate a Python function multiple times and pick the version that passes a test suite. This is one of the strongest use cases for BestOfN -- automated tests give you a perfect binary reward signal.

```python
import dspy


class GenerateFunction(dspy.Signature):
    """Write a Python function that solves the given task."""
    task_description: str = dspy.InputField(desc="What the function should do")
    function_name: str = dspy.InputField(desc="Name of the function to implement")
    code: str = dspy.OutputField(desc="Complete Python function implementation")


# --- Reward function: run tests against generated code ---

def passes_tests(args, pred):
    """Return 1.0 if the generated code passes all test cases, 0.0 otherwise."""
    test_cases = args.get("test_cases", [])
    if not test_cases:
        return 0.0

    # Execute the generated code in a sandboxed namespace
    namespace = {}
    try:
        exec(pred.code, namespace)
    except Exception:
        return 0.0

    # Run each test case
    fn = namespace.get(args["function_name"])
    if fn is None:
        return 0.0

    passed = 0
    for test_input, expected_output in test_cases:
        try:
            result = fn(*test_input) if isinstance(test_input, tuple) else fn(test_input)
            if result == expected_output:
                passed += 1
        except Exception:
            pass

    return passed / len(test_cases)


# --- Setup ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

generator = dspy.ChainOfThought(GenerateFunction)

best_generator = dspy.BestOfN(
    module=generator,
    N=3,
    reward_fn=passes_tests,
    threshold=1.0,  # Stop as soon as all tests pass
)

# --- Usage ---

result = best_generator(
    task_description="Write a function that takes a list of integers and returns a new list with duplicates removed, preserving the original order.",
    function_name="remove_duplicates",
    test_cases=[
        ([1, 2, 3, 2, 1], [1, 2, 3]),
        ([1, 1, 1], [1]),
        ([], []),
        ([5, 3, 5, 3, 5], [5, 3]),
    ],
)

print(result.code)
```

Key points:
- The reward function executes the generated code and runs test cases against it -- a fully automated, deterministic check
- `threshold=1.0` means BestOfN stops as soon as it gets code that passes all tests, potentially saving 1-2 LM calls
- The `test_cases` are passed through `args` so the reward function can access them without hardcoding
- N=3 is enough here because code generation either works or doesn't -- you don't need many samples when the pass rate per attempt is reasonable


## Example 2: Best-of-5 summarization with quality metric

Generate multiple summaries and pick the one that best covers key information while staying concise. This demonstrates a graded (non-binary) reward function.

```python
import dspy


class Summarize(dspy.Signature):
    """Produce a concise summary of the given text."""
    text: str = dspy.InputField(desc="The text to summarize")
    audience: str = dspy.InputField(desc="Who the summary is for")
    summary: str = dspy.OutputField(desc="A concise summary of the text")


# --- Reward function: multi-criteria quality score ---

def summary_quality(args, pred):
    """Score a summary on length, keyword coverage, and structure."""
    summary = pred.summary
    text = args["text"]
    words = summary.split()
    text_words = text.split()

    score = 0.0

    # 1. Length ratio: summary should be 10-25% of original length
    ratio = len(words) / max(len(text_words), 1)
    if 0.10 <= ratio <= 0.25:
        score += 0.4  # ideal compression
    elif 0.05 <= ratio <= 0.35:
        score += 0.2  # acceptable compression
    # else: too long or too short, no points

    # 2. Keyword coverage: extract frequent meaningful words from the source
    # (simple heuristic -- in production you'd use something smarter)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                 "to", "for", "of", "and", "or", "but", "with", "that", "this",
                 "it", "as", "by", "from", "be", "has", "had", "have", "not"}
    source_words = [w.lower().strip(".,!?;:") for w in text_words if len(w) > 3]
    source_words = [w for w in source_words if w not in stopwords]

    # Find the top keywords by frequency
    from collections import Counter
    word_counts = Counter(source_words)
    top_keywords = [w for w, _ in word_counts.most_common(10)]

    if top_keywords:
        summary_lower = summary.lower()
        covered = sum(1 for kw in top_keywords if kw in summary_lower)
        score += 0.4 * (covered / len(top_keywords))

    # 3. Structure: prefer summaries that start with a capital letter and end with
    # punctuation (basic well-formedness)
    if summary and summary[0].isupper() and summary.rstrip().endswith((".", "!", "?")):
        score += 0.2

    return score


# --- Setup ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

summarizer = dspy.ChainOfThought(Summarize)

best_summarizer = dspy.BestOfN(
    module=summarizer,
    N=5,
    reward_fn=summary_quality,
    threshold=0.9,  # Early stop if we get an excellent summary
)

# --- Usage ---

article = """
Artificial intelligence is transforming the healthcare industry in several
significant ways. Machine learning algorithms can now analyze medical images
with accuracy that rivals or exceeds human radiologists. Natural language
processing enables automated extraction of clinical information from
unstructured doctor notes and medical records. Predictive models help
hospitals forecast patient admissions, optimize staffing levels, and identify
patients at risk of readmission. Drug discovery has been accelerated by AI
systems that can screen millions of molecular compounds in days rather than
years. Remote patient monitoring powered by AI analyzes data from wearable
devices to detect early signs of deterioration. Despite these advances,
challenges remain around data privacy, algorithmic bias, regulatory approval,
and the need to maintain physician trust and oversight.
"""

result = best_summarizer(text=article, audience="hospital executives")
print(result.summary)
```

Key points:
- The reward function uses a weighted multi-criteria score (length + keyword coverage + structure) rather than a single binary check
- N=5 gives more candidates for subjective tasks like summarization where quality varies more across attempts
- `threshold=0.9` allows early stopping without requiring a perfect score -- useful when the metric has soft criteria
- No gold labels are needed -- the reward function evaluates quality using only the input text and the generated summary
- For production use, you could replace the keyword heuristic with a stronger signal like an LM-based judge (at the cost of extra tokens per attempt)
