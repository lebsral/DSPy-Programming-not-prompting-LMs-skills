# Ragas API Reference

> Condensed from [docs.ragas.io](https://docs.ragas.io/) and [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas). Verify against upstream for latest — Ragas API changes frequently between versions.

## Installation

```bash
pip install ragas          # evaluation only
pip install "ragas[dspy]"  # with DSPy optimizer support
```

Requires `OPENAI_API_KEY` by default. Use `llm_factory()` for other providers.

## SingleTurnSample

```python
from ragas import SingleTurnSample

sample = SingleTurnSample(
    user_input="What is DSPy?",                    # the question
    response="DSPy is a framework for...",          # the generated answer
    retrieved_contexts=["passage 1", "passage 2"],  # list[str] from retriever
    reference="DSPy is...",                         # ground truth (optional for some metrics)
)
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_input` | `str` | Yes | The input question or query |
| `response` | `str` | Yes | The generated answer |
| `retrieved_contexts` | `list[str]` | Yes | Passages returned by the retriever |
| `reference` | `str \| None` | For some metrics | Ground-truth answer |

## EvaluationDataset

```python
from ragas import EvaluationDataset

dataset = EvaluationDataset(samples=[sample1, sample2, ...])
```

## evaluate()

```python
from ragas import evaluate

result = evaluate(
    dataset=dataset,            # EvaluationDataset -- required
    metrics=[metric1, metric2], # list of metric instances -- required
    llm=None,                   # custom LLM (default: OpenAI)
)
```

**Returns:** `EvaluationResult` with:
- Dict-like access: `result["faithfulness"]` -- aggregate score
- `result.to_pandas()` -- per-sample DataFrame

## Available Metrics

| Metric | Import | Needs reference? | Evaluates |
|--------|--------|:----------------:|-----------|
| `Faithfulness()` | `from ragas.metrics import Faithfulness` | No | Generator -- is answer grounded in context? |
| `AnswerRelevancy()` | `from ragas.metrics import AnswerRelevancy` | No | Generator -- does answer address the question? |
| `ContextPrecision()` | `from ragas.metrics import ContextPrecision` | Yes | Retriever -- are relevant docs ranked higher? |
| `ContextRecall()` | `from ragas.metrics import ContextRecall` | Yes | Retriever -- did retrieval find all relevant info? |
| `AnswerCorrectness()` | `from ragas.metrics import AnswerCorrectness` | Yes | End-to-end -- does answer match reference? |

All metrics return scores in the range 0.0 to 1.0 (higher is better).

## Custom LLM

```python
from ragas.llms import llm_factory

# Use any LiteLLM-supported provider string
evaluator_llm = llm_factory("anthropic/claude-sonnet-4-5-20250929")
result = evaluate(dataset=dataset, metrics=[Faithfulness()], llm=evaluator_llm)
```

**Deprecated:** `LangchainLLMWrapper` was removed in Ragas v0.3.8+. Use `llm_factory()` instead.

## DSPyOptimizer

```python
from ragas.integrations.dspy import DSPyOptimizer

optimizer = DSPyOptimizer(metric=Faithfulness())
optimized_metric = optimizer.optimize(dataset=labeled_eval_dataset)
```

Requires `pip install "ragas[dspy]"`. Uses MIPROv2 internally to optimize the metric's own prompts.

## Version History (breaking changes)

| Version | Key change |
|---------|-----------|
| v0.2 | Introduced `EvaluationDataset`, `SingleTurnSample` (replaced `datasets.Dataset`) |
| v0.3.8 | Deprecated `LangchainLLMWrapper`, `LlamaIndexLLMWrapper` |
| v0.3.9 | Deprecated `ground_truths` parameter, removed AspectCritic/SimpleCriteria |
| v0.4.0 | Migrated metrics to BasePrompt architecture, added Instructor + LiteLLM adapters |
