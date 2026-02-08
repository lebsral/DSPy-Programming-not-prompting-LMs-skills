# Database Querying Examples

## Example 1: E-commerce analytics assistant

A data assistant that answers business questions about an e-commerce database.

### Schema

```sql
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    signup_date DATE,
    plan VARCHAR(20)  -- 'free', 'pro', 'enterprise'
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    total_amount DECIMAL(10, 2),
    status VARCHAR(20),  -- 'pending', 'completed', 'refunded'
    created_at TIMESTAMP
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10, 2)
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER,
    unit_price DECIMAL(10, 2)
);
```

### Full implementation

```python
import dspy
from sqlalchemy import create_engine, inspect, text

# Setup
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

engine = create_engine("postgresql://ai_reader:pass@localhost:5432/ecommerce")

# Schema with business context
TABLE_DESCRIPTIONS = {
    "customers": "Customer profiles with signup info and subscription plan",
    "orders": "Purchase orders with totals and status tracking",
    "products": "Product catalog with categories and prices",
    "order_items": "Line items linking orders to products with quantities",
}

schema = get_enriched_schema(engine, table_descs=TABLE_DESCRIPTIONS)

# Build pipeline
class EcommerceQA(dspy.Module):
    def __init__(self, engine, schema):
        self.engine = engine
        self.schema = schema
        self.generate_sql = dspy.ChainOfThought(GenerateSQL)
        self.interpret = dspy.ChainOfThought(InterpretResults)

    def forward(self, question):
        result = self.generate_sql(schema=self.schema, question=question)
        sql = result.sql.strip().rstrip(";")
        validate_sql(sql)
        rows = execute_query(self.engine, sql)
        interpretation = self.interpret(
            question=question, sql=sql, results=str(rows[:20])
        )
        return dspy.Prediction(sql=sql, rows=rows, answer=interpretation.answer)

qa = EcommerceQA(engine, schema)

# Test it
result = qa(question="What were our top 5 products by revenue last month?")
print(result.sql)
# SELECT p.name, SUM(oi.quantity * oi.unit_price) AS revenue
# FROM order_items oi
# JOIN orders o ON oi.order_id = o.id
# JOIN products p ON oi.product_id = p.id
# WHERE o.created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
#   AND o.created_at < DATE_TRUNC('month', CURRENT_DATE)
#   AND o.status = 'completed'
# GROUP BY p.name
# ORDER BY revenue DESC
# LIMIT 5

print(result.answer)
# "The top 5 products by revenue last month were:
# 1. Premium Widget — $12,450
# 2. Pro Gadget — $8,320
# ..."
```

### Training data for optimization

```python
trainset = [
    dspy.Example(
        question="How many customers signed up this week?",
        answer="23 customers signed up this week.",
    ).with_inputs("question"),
    dspy.Example(
        question="What's the average order value?",
        answer="The average order value is $67.50.",
    ).with_inputs("question"),
    dspy.Example(
        question="Which category has the most products?",
        answer="Electronics has the most products with 45 items.",
    ).with_inputs("question"),
    dspy.Example(
        question="How many orders were refunded last month?",
        answer="There were 12 refunded orders last month.",
    ).with_inputs("question"),
    dspy.Example(
        question="Who are our top 3 customers by total spending?",
        answer="The top 3 customers by spending are: 1. Acme Corp ($15,200), 2. TechStart ($12,800), 3. DataFlow ($9,400).",
    ).with_inputs("question"),
]
```

## Example 2: HR data assistant with table selection

For a larger database where table selection matters.

### Schema (10+ tables)

```python
TABLE_DESCRIPTIONS = {
    "employees": "Employee profiles with name, department, role, hire date",
    "departments": "Department list with managers and budgets",
    "salaries": "Salary history with effective dates",
    "time_off": "PTO requests with approval status",
    "performance_reviews": "Annual performance reviews with ratings 1-5",
    "benefits": "Employee benefit enrollments (health, dental, 401k)",
    "training": "Training courses completed by employees",
    "positions": "Open and filled job positions",
    "expenses": "Employee expense reports",
    "office_locations": "Office locations with addresses and capacity",
}
```

### Pipeline with table selection

```python
class HRQA(dspy.Module):
    def __init__(self, engine, schema):
        self.engine = engine
        self.schema = schema
        self.select_tables = dspy.ChainOfThought(SelectTables)
        self.generate_sql = dspy.ChainOfThought(GenerateSQL)
        self.interpret = dspy.ChainOfThought(InterpretResults)

    def forward(self, question):
        # Step 1: pick relevant tables
        selected = self.select_tables(schema=self.schema, question=question)
        focused_schema = filter_schema(self.schema, selected.tables)

        # Step 2: generate SQL with focused schema
        result = self.generate_sql(schema=focused_schema, question=question)
        sql = result.sql.strip().rstrip(";")
        validate_sql(sql)

        # Step 3: execute and interpret
        rows = execute_query(self.engine, sql)
        interpretation = self.interpret(
            question=question, sql=sql, results=str(rows[:20])
        )
        return dspy.Prediction(
            sql=sql, tables=selected.tables,
            rows=rows, answer=interpretation.answer,
        )

qa = HRQA(engine, schema)

# Test
result = qa(question="What's the average salary by department?")
print(result.tables)
# ["employees", "departments", "salaries"]

print(result.sql)
# SELECT d.name AS department, AVG(s.amount) AS avg_salary
# FROM salaries s
# JOIN employees e ON s.employee_id = e.id
# JOIN departments d ON e.department_id = d.id
# WHERE s.effective_date = (
#     SELECT MAX(effective_date) FROM salaries WHERE employee_id = e.id
# )
# GROUP BY d.name
# ORDER BY avg_salary DESC

print(result.answer)
# "Average salary by department:
# - Engineering: $145,000
# - Product: $132,000
# - Marketing: $118,000
# ..."
```

### Optimizing with MIPROv2

```python
trainset = [
    dspy.Example(
        question="How many employees are in engineering?",
        answer="There are 42 employees in the Engineering department.",
    ).with_inputs("question"),
    dspy.Example(
        question="Who has the most PTO days remaining?",
        answer="Sarah Chen has 18 PTO days remaining.",
    ).with_inputs("question"),
    dspy.Example(
        question="What's the average performance rating this year?",
        answer="The average performance rating this year is 3.7 out of 5.",
    ).with_inputs("question"),
    # ... 20-50 examples covering common HR questions
]

optimizer = dspy.MIPROv2(metric=answer_quality, auto="medium")
optimized = optimizer.compile(qa, trainset=trainset)
optimized.save("optimized_hr_qa.json")
```
