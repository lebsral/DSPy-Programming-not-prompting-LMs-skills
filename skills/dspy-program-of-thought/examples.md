# dspy-program-of-thought -- Worked Examples

## Example 1: Financial calculation

Compute compound interest with monthly contributions -- the kind of calculation where LMs routinely make mistakes but code gets right every time.

```python
import dspy


class InvestmentCalculator(dspy.Signature):
    """Calculate the future value of an investment given the parameters."""
    initial_deposit: float = dspy.InputField(desc="Starting amount in dollars")
    monthly_contribution: float = dspy.InputField(desc="Amount added each month")
    annual_rate: float = dspy.InputField(desc="Annual interest rate as a decimal, e.g. 0.07 for 7%")
    years: int = dspy.InputField(desc="Number of years")
    future_value: float = dspy.OutputField(desc="Total value at the end, rounded to 2 decimal places")


lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

calc = dspy.ProgramOfThought(InvestmentCalculator)

result = calc(
    initial_deposit=10000.0,
    monthly_contribution=500.0,
    annual_rate=0.07,
    years=20,
)
print(f"Future value: ${result.future_value:,.2f}")
# The generated code computes compound interest with monthly compounding
# and contributions -- exact to the penny.
```

Key points:
- Typed signature fields (`float`, `int`) guide the LM to produce code that returns the right type
- The desc fields tell the LM what each parameter means, so the generated code uses them correctly
- Compound interest with contributions is a multi-step formula that LMs frequently get wrong in natural language reasoning


## Example 2: Data analysis with computation

Analyze sales data to compute aggregates, rankings, and derived metrics. This is the kind of task where you would normally reach for a spreadsheet.

```python
import dspy


class SalesAnalysis(dspy.Signature):
    """Analyze the sales data and answer the question with a computed result."""
    sales_data: str = dspy.InputField(desc="Sales data as a text table or JSON string")
    question: str = dspy.InputField(desc="An analytical question about the data")
    answer: str = dspy.OutputField(desc="The computed answer")


lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

analyzer = dspy.ProgramOfThought(SalesAnalysis)

sales_csv = """
Region,Q1,Q2,Q3,Q4
North,120000,145000,132000,168000
South,98000,103000,115000,125000
East,156000,142000,138000,171000
West,87000,95000,102000,118000
""".strip()

# Question 1: Aggregation
result = analyzer(
    sales_data=sales_csv,
    question="Which region had the highest total annual sales, and what was the total?",
)
print(result.answer)

# Question 2: Growth analysis
result = analyzer(
    sales_data=sales_csv,
    question="What is the quarter-over-quarter growth rate for each region in Q4 vs Q3? Return as percentages.",
)
print(result.answer)

# Question 3: Ranking
result = analyzer(
    sales_data=sales_csv,
    question="Rank the quarters by total sales across all regions, from highest to lowest.",
)
print(result.answer)
```

Key points:
- The LM generates code to parse the CSV, compute aggregates, and format the result
- Each question produces different code -- the LM adapts its computation to the question
- This avoids the common problem of LMs miscounting or misadding numbers in tables


## Example 3: Date/time reasoning

Date calculations involve edge cases (leap years, month lengths, timezone offsets) that trip up natural language reasoning. Code handles them correctly via Python's `datetime` module.

```python
import dspy


class DateCalculator(dspy.Signature):
    """Solve date and time problems by computing the answer."""
    question: str = dspy.InputField(desc="A question involving dates, times, or durations")
    answer: str = dspy.OutputField(desc="The computed answer")


lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

solver = dspy.ProgramOfThought(DateCalculator)

# Leap year awareness
result = solver(
    question="How many days are there between February 15, 2024 and March 15, 2024?"
)
print(result.answer)  # 29 (2024 is a leap year)

# Business day calculation
result = solver(
    question="If a project starts on Monday, January 6, 2025 and takes 45 business days "
             "(excluding weekends), what date does it end?"
)
print(result.answer)

# Age calculation with edge cases
result = solver(
    question="Someone was born on February 29, 2000. How old are they on March 1, 2025? "
             "Give the answer in years and days."
)
print(result.answer)

# Duration between timestamps
result = solver(
    question="A server went down at 2025-01-15 23:47:12 UTC and came back at "
             "2025-01-16 02:13:45 UTC. How long was the outage in hours and minutes?"
)
print(result.answer)
```

Key points:
- Python's `datetime` module handles leap years, month boundaries, and weekday logic correctly
- Business day calculations require looping and weekday checks -- natural language reasoning almost always miscounts
- The LM generates different code for each question type (duration, business days, age calculation)
- No external libraries needed -- `datetime` is in the standard library and always available in the sandbox
