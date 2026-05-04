# TwoStepAdapter API Reference

> Condensed from [dspy.ai/api/adapters/TwoStepAdapter](https://dspy.ai/api/adapters/TwoStepAdapter/). Verify against upstream for latest.

## Constructor

```python
dspy.TwoStepAdapter(
    main_lm,         # dspy.LM -- the reasoning model
    extraction_lm,   # dspy.LM -- the parsing/extraction model
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `main_lm` | `dspy.LM` | Primary reasoning model (o1, o3, DeepSeek-R1, etc.) |
| `extraction_lm` | `dspy.LM` | Fast model that extracts structured fields from main_lm output |

## How it works

1. **Phase 1:** Sends the prompt to `main_lm` without formatting constraints. The reasoning model generates freely.
2. **Phase 2:** Takes the raw output from Phase 1 and sends it to `extraction_lm` with ChatAdapter formatting, asking it to extract the required fields.

The user sees a single call -- the two-phase flow is transparent.

## Configuration

### Global (all modules use TwoStepAdapter)

```python
main_lm = dspy.LM("openai/o3-mini")
extraction_lm = dspy.LM("openai/gpt-4o-mini")
adapter = dspy.TwoStepAdapter(main_lm=main_lm, extraction_lm=extraction_lm)

dspy.configure(lm=main_lm, adapter=adapter)
```

### Per-module (only specific modules use TwoStepAdapter)

```python
module.lm = reasoning_lm
module.adapter = dspy.TwoStepAdapter(
    main_lm=reasoning_lm,
    extraction_lm=extraction_lm,
)
```

## Supported reasoning models

| Model | Provider string | Notes |
|-------|----------------|-------|
| o1 | `openai/o1` | Strips system prompts |
| o3 | `openai/o3` | Highest capability |
| o3-mini | `openai/o3-mini` | Cost-effective |
| DeepSeek-R1 | `deepseek/deepseek-r1` | Open-weight |
| Claude (extended thinking) | `anthropic/claude-sonnet-4-5-20250929` | Needs `thinking` param |

## Extended thinking configuration

```python
thinking_lm = dspy.LM(
    "anthropic/claude-sonnet-4-5-20250929",
    thinking={"type": "enabled", "budget_tokens": 10000},
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `thinking.type` | `str` | Must be `"enabled"` |
| `thinking.budget_tokens` | `int` | Max tokens for thinking (1000-100000) |

## Comparison with other adapters

| Adapter | Best for | Structured output |
|---------|----------|-------------------|
| `ChatAdapter` | Standard chat models (GPT-4o, Claude, Gemini) | Direct -- model follows format |
| `JSONAdapter` | Models with native JSON mode | JSON parsing |
| `TwoStepAdapter` | Reasoning models (o1, o3, R1) | Two-phase -- generate then extract |
