---
name: ai-querying-databases
description: Build AI that answers questions about your database. Use when you need text-to-SQL, natural language database queries, a data assistant for non-technical users, AI-powered analytics, plain English database search, or a chatbot that talks to your database. Also used for text-to-SQL that actually works, AI SQL generation is unreliable, let non-technical users query data, build a data analyst chatbot, business intelligence with AI, self-service analytics, AI dashboard queries, ask questions about my database in English, SQL copilot, AI-powered data exploration, Metabase alternative with AI, chat with your Postgres, natural language analytics, data chatbot for stakeholders, DSPy pipelines for schema understanding and SQL generation.
---

# Build AI That Answers Questions About Your Database

Guide the user through building an AI that takes plain English questions and returns answers from a SQL database. The pattern: understand the schema, generate SQL, validate it, run it, and explain the results.

## When you need this

- Sales reps asking "how many deals closed last month?" without writing SQL
- Executives asking revenue questions in plain English
- Support agents looking up customer records by description
- Internal data assistants for non-technical staff
- Any "chat with your database" feature

## How it's different from document search

| | Document search (`/ai-searching-docs`) | Database querying (this skill) |
|---|---|---|
| Data type | Unstructured text (PDFs, articles, docs) | Structured data (tables, rows, columns) |
| How it works | Embed + retrieve passages | Understand schema + generate SQL |
| Output | Text answer grounded in passages | Data from query results + interpretation |
| Key challenge | Finding relevant passages | Writing correct, safe SQL |

## Step 1: Understand the setup

Ask the user:
1. **What database?** (Postgres, MySQL, SQLite, Snowflake, BigQuery, etc.)
2. **What tables matter?** (all of them, or a subset?)
3. **Who asks questions?** (technical users, business users, customers?)
4. **Read-only access?** (this should always be yes for AI-generated SQL)

## Step 2: Connect to your database

Use SQLAlchemy for provider-agnostic database access:

```python
from sqlalchemy import create_engine, inspect, text

# PostgreSQL
engine = create_engine("postgresql://user:pass@host:5432/mydb")

# MySQL
engine = create_engine("mysql+pymysql://user:pass@host:3306/mydb")

# SQLite (for development)
engine = create_engine("sqlite:///local.db")

# Snowflake
engine = create_engine("snowflake://user:pass@account/db/schema")

# BigQuery
engine = create_engine("bigquery://project/dataset")
```

### Build schema descriptions for the AI

The AI needs to understand your tables to write correct SQL:

```python
def get_schema_description(engine, tables=None):
    """Build a text description of database schema for the AI."""
    inspector = inspect(engine)
    tables = tables or inspector.get_table_names()

    descriptions = []
    for table in tables:
        columns = inspector.get_columns(table)
        col_descs = []
        for col in columns:
            col_descs.append(f"  - {col['name']} ({col['type']})")

        pk = inspector.get_pk_constraint(table)
        pk_cols = pk['constrained_columns'] if pk else []

        desc = f"Table: {table}\n"
        if pk_cols:
            desc += f"  Primary key: {', '.join(pk_cols)}\n"
        desc += "  Columns:\n" + "\n".join(col_descs)
        descriptions.append(desc)

    return "\n\n".join(descriptions)

schema = get_schema_description(engine)
print(schema)
```

### Add business context (optional but helpful)

Raw column names like `cust_ltv_90d` don't mean much to the AI. Add descriptions:

```python
TABLE_DESCRIPTIONS = {
    "orders": "Customer orders with amounts, dates, and status",
    "customers": "Customer profiles with contact info and signup date",
    "products": "Product catalog with names, prices, and categories",
}

COLUMN_DESCRIPTIONS = {
    "orders.cust_ltv_90d": "Customer lifetime value over the last 90 days in USD",
    "orders.gmv": "Gross merchandise value (total order amount before discounts)",
}

def get_enriched_schema(engine, table_descs=None, col_descs=None):
    """Schema description with business context."""
    inspector = inspect(engine)
    table_descs = table_descs or {}
    col_descs = col_descs or {}

    descriptions = []
    for table in inspector.get_table_names():
        desc = f"Table: {table}"
        if table in table_descs:
            desc += f" -- {table_descs[table]}"
        desc += "\n  Columns:\n"

        for col in inspector.get_columns(table):
            col_key = f"{table}.{col['name']}"
            col_desc = f"  - {col['name']} ({col['type']})"
            if col_key in col_descs:
                col_desc += f" -- {col_descs[col_key]}"
            desc += col_desc + "\n"

        descriptions.append(desc)

    return "\n".join(descriptions)
```

## Step 3: Build the text-to-SQL pipeline

Two-stage approach: first pick the relevant tables, then generate SQL.

### Hard validation gate

The modules below call `validate_sql(sql)` to reject unsafe SQL before execution. It enforces the same hard constraints as the `sql_safety_reward` function in Step 4, but raises instead of scoring — use it inside `forward` so a single bad query never reaches the database:

```python
def validate_sql(sql: str) -> str:
    """Raise ValueError if the SQL is unsafe; otherwise return it unchanged."""
    sql_clean = sql.strip().rstrip(";")
    sql_upper = sql_clean.upper()

    if not sql_upper.startswith("SELECT"):
        raise ValueError(f"Only SELECT queries are allowed: {sql}")

    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "EXEC"]
    for keyword in dangerous:
        if keyword in sql_upper.split("SELECT", 1)[0]:
            raise ValueError(f"Unsafe SQL rejected ({keyword}): {sql}")

    return sql_clean
```

### Stage 1: Table selection (for databases with many tables)

```python
import dspy

class SelectTables(dspy.Signature):
    """Given a database schema and a user question, select which tables
    are needed to answer the question."""
    schema: str = dspy.InputField(desc="Database schema description")
    question: str = dspy.InputField(desc="User's question in plain English")
    tables: list[str] = dspy.OutputField(desc="List of table names needed")
```

### Stage 2: SQL generation

```python
class GenerateSQL(dspy.Signature):
    """Write a SQL SELECT query to answer the user's question.
    Only use tables and columns that exist in the schema."""
    schema: str = dspy.InputField(desc="Database schema for relevant tables")
    question: str = dspy.InputField(desc="User's question in plain English")
    sql: str = dspy.OutputField(desc="SQL SELECT query (read-only, no mutations)")
```

### The full pipeline

```python
class DatabaseQA(dspy.Module):
    def __init__(self, engine, schema, use_table_selection=False):
        self.engine = engine
        self.full_schema = schema
        self.use_table_selection = use_table_selection

        if use_table_selection:
            self.select_tables = dspy.ChainOfThought(SelectTables)
        self.generate_sql = dspy.ChainOfThought(GenerateSQL)
        self.interpret = dspy.ChainOfThought(InterpretResults)

    def forward(self, question):
        # Pick relevant tables (for large schemas)
        if self.use_table_selection:
            selected = self.select_tables(
                schema=self.full_schema, question=question
            )
            schema = filter_schema(self.full_schema, selected.tables)
        else:
            schema = self.full_schema

        # Generate SQL
        result = self.generate_sql(schema=schema, question=question)
        sql = result.sql.strip().rstrip(";")

        # Validate (see Step 4)
        validate_sql(sql)

        # Execute
        rows = execute_query(self.engine, sql)

        # Interpret results
        interpretation = self.interpret(
            question=question, sql=sql, results=str(rows[:20])
        )
        return dspy.Prediction(
            sql=sql, rows=rows, answer=interpretation.answer
        )
```

### Helper: filter schema to selected tables

```python
def filter_schema(full_schema, table_names):
    """Keep only the schema sections for selected tables."""
    sections = full_schema.split("\n\n")
    filtered = []
    for section in sections:
        for table in table_names:
            if section.startswith(f"Table: {table}"):
                filtered.append(section)
                break
    return "\n\n".join(filtered)
```

## Step 4: Validate SQL before execution

Never run AI-generated SQL without validation. Use a reward function with `dspy.Refine` to enforce hard safety constraints and penalize style issues:

```python
import sqlparse

def sql_safety_reward(args, pred):
    """Reward function for SQL safety and correctness. Returns 0.0-1.0."""
    sql = pred.sql.strip().rstrip(";") if hasattr(pred, "sql") else ""
    sql_upper = sql.upper()
    score = 1.0

    # Hard safety constraints -- fail immediately if violated
    if not sql_upper.startswith("SELECT"):
        return 0.0

    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "EXEC"]
    for keyword in dangerous:
        if keyword in sql_upper.split("SELECT", 1)[0]:
            return 0.0

    # Syntax check
    parsed = sqlparse.parse(sql)
    if len(parsed) != 1 or parsed[0].get_type() != "SELECT":
        return 0.0

    # Style penalty (soft) -- prefer explicit JOIN ... ON syntax
    if "JOIN" in sql_upper and "ON" not in sql_upper:
        score -= 0.1

    return score


# Wrap the SQL generation step with Refine
def make_validated_sql_module(engine, schema):
    generate_sql = dspy.ChainOfThought(GenerateSQL)

    return dspy.Refine(
        module=generate_sql,
        N=3,
        reward_fn=sql_safety_reward,
        threshold=0.9,
    )
```

### Execute with safety limits

```python
from sqlalchemy import text

def execute_query(engine, sql, row_limit=100, timeout_seconds=30):
    """Execute validated SQL with safety limits."""
    # Add row limit if not present
    if "LIMIT" not in sql.upper():
        sql = f"{sql} LIMIT {row_limit}"

    with engine.connect() as conn:
        conn = conn.execution_options(timeout=timeout_seconds)
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = [dict(zip(columns, row)) for row in result.fetchall()]

    return rows
```

## Step 5: Interpret results

Convert raw query results back to a natural language answer:

```python
class InterpretResults(dspy.Signature):
    """Convert SQL query results into a clear, natural language answer
    to the user's original question."""
    question: str = dspy.InputField(desc="The user's original question")
    sql: str = dspy.InputField(desc="The SQL query that was run")
    results: str = dspy.InputField(desc="Query results as a string")
    answer: str = dspy.OutputField(desc="Natural language answer to the question")
```

## Step 6: Handle large schemas

For databases with 50+ tables, sending the full schema to the AI is expensive and confusing. Use embedding-based schema retrieval instead:

1. Build a ChromaDB index of table descriptions at startup (`pip install chromadb`)
2. At query time, embed the user's question and retrieve the top-k most relevant table schemas
3. Pass only those table schemas to `GenerateSQL`

The two-stage `SelectTables` module (Step 3) is a lighter alternative when you have 10–50 tables — it uses the LM itself to pick relevant tables rather than embeddings. For schemas over 50 tables, use the ChromaDB approach to avoid token overload.

See [examples.md](examples.md) for the full `SchemaRetriever` and `LargeSchemaQA` implementations with ChromaDB.

## Step 7: Test and optimize

### SQL execution accuracy metric

```python
def sql_accuracy(example, prediction, trace=None):
    """Check if the generated SQL returns the correct answer."""
    try:
        # Compare results (not SQL text — many valid SQL queries per question)
        expected = set(str(r) for r in example.expected_rows)
        actual = set(str(r) for r in prediction.rows)
        return float(expected == actual)
    except Exception:
        return 0.0

def answer_quality(example, prediction, trace=None):
    """Check if the natural language answer is correct."""
    judge = dspy.Predict("question, expected_answer, predicted_answer -> is_correct: bool")
    result = judge(
        question=example.question,
        expected_answer=example.answer,
        predicted_answer=prediction.answer,
    )
    return float(result.is_correct)
```

### Build training data

```python
trainset = [
    dspy.Example(
        question="How many orders were placed last month?",
        answer="There were 1,247 orders placed last month.",
        expected_sql="SELECT COUNT(*) FROM orders WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')",
    ).with_inputs("question"),
    # Add 20-50 question/answer pairs covering your common queries
]
```

### Optimize

```python
optimizer = dspy.MIPROv2(metric=answer_quality, auto="medium")
optimized = optimizer.compile(DatabaseQA(engine, schema), trainset=trainset)
optimized.save("optimized_db_qa.json")
```

## Step 8: Security and production

### Security checklist

| Control | How |
|---------|-----|
| Read-only database user | `GRANT SELECT ON ALL TABLES TO ai_reader` |
| Query timeout | `execution_options(timeout=30)` in SQLAlchemy |
| Row limit | Always append `LIMIT` to queries |
| Table allowlist | Only include permitted tables in the schema |
| SQL validation | `dspy.Refine` with a safety reward function for SELECT-only, no dangerous keywords |
| Audit logging | Log every question, generated SQL, and results |
| No raw credentials | Use environment variables or secrets manager |

### Audit logging

```python
import json
from datetime import datetime

def log_query(question, sql, row_count, user_id=None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "question": question,
        "sql": sql,
        "row_count": row_count,
    }
    with open("query_audit.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
```

### Table allowlist

Pass only the permitted table names to `get_schema_description(engine, tables=list(ALLOWED_TABLES))`. Never include PII tables (e.g., raw payment info, SSNs) in the schema fed to the AI.

## When NOT to use text-to-SQL

Text-to-SQL is not the right tool in every case:

- **Schema changes frequently** — if tables or column names change more than weekly, the AI's understanding drifts fast. Invest in a schema refresh pipeline before adding AI queries.
- **Users need complex joins across 10+ tables** — current LMs struggle with JOINs that span many foreign keys. A better approach is pre-built semantic views the AI can query.
- **Sub-second latency required** — the generate-SQL-validate-execute-interpret chain adds 2–5 seconds per query. Cache common queries or use a hybrid (SQL templates + AI parameter filling) for latency-sensitive paths.
- **High-stakes writes needed** — if business logic requires INSERT/UPDATE based on AI reasoning, text-to-SQL is dangerous. Use a structured form or workflow instead; AI generates queries only for reads.

## Key patterns

- **Two-stage pipeline**: table selection + SQL generation works better than one giant prompt
- **Validate before executing**: never run AI-generated SQL without safety checks
- **Compare results, not SQL**: many valid SQL queries produce the same answer
- **Business context matters**: column descriptions improve accuracy more than extra examples
- **Start with a small table allowlist**: expand as you build confidence
- **Read-only, always**: the AI database user should never have write permissions

## Gotchas

- **Do not declare `reasoning` in signatures wrapped with `dspy.ChainOfThought`** — ChainOfThought automatically prepends a `reasoning` field to your signature. If you also declare `reasoning: str = dspy.OutputField()` in the signature class, ChainOfThought injects a second one. The result is unpredictable output. Omit `reasoning` from your signature and let ChainOfThought handle it. Use `dspy.Predict` if you want a signature without an injected reasoning field.

- **LMs return SQL in markdown code fences** — models often wrap output in triple-backtick blocks (`\`\`\`sql ... \`\`\``). The `validate_sql` function above handles semicolons but not markdown fences. Strip them before passing to validate: `re.sub(r'^```\w*\n?|```$', '', sql.strip()).strip()`. Add this to your `forward()` before `validate_sql()`.

- **Schema descriptions built at init go stale** — if you call `get_schema_description(engine)` once at startup and store it, newly added columns or tables are invisible to the AI until restart. Build schema descriptions per request (cheap for small schemas) or use a short-TTL cache.

- **Compare result rows, not SQL text** — training data built around `expected_sql` is fragile: `SELECT COUNT(*) FROM orders WHERE ...` and `SELECT COUNT(id) FROM orders WHERE ...` return the same answer but fail string comparison. The `sql_accuracy` metric above compares result sets — use that pattern for all evals and metric functions.

- **Appending `LIMIT` after subqueries breaks SQL** — the `execute_query` helper appends `LIMIT 100` if no LIMIT exists. This corrupts queries like `SELECT * FROM (SELECT ... ORDER BY ...) AS sub` where the limit belongs inside the subquery. Check for a subquery before blindly appending: only append LIMIT to flat `SELECT ... FROM table` patterns.

## Cross-references

> Install any skill: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill <name>`

- `/dspy-refine` for the retry-with-feedback pattern used in SQL validation
- `/dspy-signatures` for defining input/output contracts for SelectTables, GenerateSQL, InterpretResults
- `/dspy-chain-of-thought` for the ChainOfThought reasoning injection pattern
- `/ai-serving-apis` to put your database assistant behind a REST API
- `/ai-building-pipelines` for complex multi-step query workflows
- `/ai-checking-outputs` for additional SQL validation patterns
- `/ai-following-rules` to enforce query policies (e.g., no queries on PII columns)
- `/ai-improving-accuracy` to measure and optimize query quality
- `/ai-tracing-requests` to debug individual query failures
- **Install `/ai-do` if you do not have it** — it routes any AI problem to the right skill and is the fastest way to work: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill ai-do`

## Additional resources

- For worked examples and the full large-schema ChromaDB implementation, see [examples.md](examples.md)
- For DSPy API signatures and parameter tables, see [reference.md](reference.md)
