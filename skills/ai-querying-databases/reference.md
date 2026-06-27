> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Text-to-SQL

## Signatures

The three signatures used by the pipeline — docstring becomes the task instruction:

```python
class SelectTables(dspy.Signature):
    """Select which tables are needed to answer the question."""
    schema: str = dspy.InputField(desc="Database schema description")
    question: str = dspy.InputField(desc="User's question in plain English")
    tables: list[str] = dspy.OutputField(desc="List of table names needed")
    reasoning: str = dspy.OutputField(desc="Why these tables are needed")

class GenerateSQL(dspy.Signature):
    """Write a SQL SELECT query. Only use tables and columns in the schema."""
    schema: str = dspy.InputField(desc="Database schema for relevant tables")
    question: str = dspy.InputField(desc="User's question in plain English")
    sql: str = dspy.OutputField(desc="SQL SELECT query (read-only, no mutations)")

class InterpretResults(dspy.Signature):
    """Convert SQL results into a clear, natural language answer."""
    question: str = dspy.InputField(desc="The user's original question")
    sql: str = dspy.InputField(desc="The SQL query that was run")
    results: str = dspy.InputField(desc="Query results as a string")
    answer: str = dspy.OutputField(desc="Natural language answer to the question")
```

### dspy.InputField / dspy.OutputField

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desc` | `str` | `""` | Describes the field to the LM |
| `prefix` | `str \| None` | `None` | Label shown in the prompt |

## dspy.ChainOfThought

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Defines inputs/outputs |
| `rationale_field` | `FieldInfo \| None` | `None` | Custom reasoning field |

Injects a `reasoning` field before outputs automatically — do not declare it in your signature. Used for all three stages: table selection, SQL generation, and result interpretation.

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold=None, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | Module to wrap |
| `N` | `int` | required | Max retry attempts |
| `reward_fn` | `Callable[[args, pred], float]` | required | Returns 0.0–1.0 |
| `threshold` | `float \| None` | `None` | Stop early if score exceeds this |

Wrap SQL generation so unsafe queries are retried before execution:

```python
validated_sql = dspy.Refine(
    module=dspy.ChainOfThought(GenerateSQL),
    N=3,
    reward_fn=sql_safety_reward,
    threshold=0.9,
)
```

> `dspy.Assert` and `dspy.Suggest` were removed in DSPy 3.x. Use `dspy.Refine` instead.

## dspy.Predict

No reasoning step. Use for the LM judge inside metrics:
`dspy.Predict("question, expected_answer, predicted_answer -> is_correct: bool")`

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
dspy.MIPROv2(metric, auto='light', max_bootstrapped_demos=4,
             max_labeled_demos=4, num_candidates=None, num_threads=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Scoring function `(example, pred, trace) -> float` |
| `auto` | `'light' \| 'medium' \| 'heavy'` | `'light'` | Optimization intensity |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `4` | Max labeled demos from trainset |

```python
optimizer = dspy.MIPROv2(metric=answer_quality, auto="medium")
optimized = optimizer.compile(DatabaseQA(engine, schema), trainset=trainset)
optimized.save("optimized_db_qa.json")
```

## Quick Reference

### Package versions

| Package | Purpose | Install |
|---------|---------|---------|
| `dspy` ≥ 3.2.1 | Core framework | `pip install -U dspy` |
| `sqlalchemy` ≥ 2.0 | Database connections | `pip install sqlalchemy` |
| `sqlparse` ≥ 0.5 | SQL syntax validation | `pip install sqlparse` |
| `chromadb` ≥ 0.5 | Schema vector index (large DBs) | `pip install chromadb` |

### LM configuration

```python
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # any LiteLLM-supported provider
```

### SQLAlchemy connection strings

| Database | Connection string |
|----------|------------------|
| PostgreSQL | `postgresql://user:pass@host:5432/db` |
| MySQL | `mysql+pymysql://user:pass@host:3306/db` |
| SQLite | `sqlite:///local.db` |
| Snowflake | `snowflake://user:pass@account/db/schema` |
| BigQuery | `bigquery://project/dataset` |

### Reward function signature

`reward_fn(args: dict, pred: dspy.Prediction) -> float` — return 0.0–1.0.

### Pipeline at a glance

```
question
  → ChainOfThought(SelectTables)        # optional, for 50+ table schemas
  → Refine(ChainOfThought(GenerateSQL)) # retries until safety reward ≥ threshold
  → validate_sql()                      # raises ValueError on unsafe SQL
  → execute_query()                     # read-only user, LIMIT enforced
  → ChainOfThought(InterpretResults)
  → dspy.Prediction(sql, rows, answer)
```
