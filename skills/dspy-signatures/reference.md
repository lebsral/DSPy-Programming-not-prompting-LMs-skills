# Signatures API Reference

> Condensed from [dspy.ai/api/signatures](https://dspy.ai/api/signatures/Signature). Verify against upstream for latest.

## dspy.Signature

Base class for defining LM call contracts. Subclass it to create typed signatures.

```python
class MySignature(dspy.Signature):
    """Task instruction goes here."""
    input_field: str = dspy.InputField(desc="description")
    output_field: str = dspy.OutputField(desc="description")
```

### Class Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `with_instructions` | `(instructions: str) -> type[Signature]` | Return new Signature with replaced instructions |
| `append` | `(name: str, field, type_=None) -> type[Signature]` | Add field at end of inputs/outputs |
| `prepend` | `(name: str, field, type_=None) -> type[Signature]` | Insert field at position 0 |
| `insert` | `(index: int, name: str, field, type_=None) -> type[Signature]` | Insert field at specific index |
| `delete` | `(name: str) -> type[Signature]` | Remove a field (no error if absent) |
| `with_updated_fields` | `(name: str, type_=None, **kwargs) -> type[Signature]` | Update field metadata |
| `equals` | `(other) -> bool` | Compare JSON schemas |

All methods are non-mutating -- they return new Signature classes.

## dspy.InputField

```python
dspy.InputField(desc=None, prefix=None, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desc` | `str \| None` | `None` | Natural-language description shown to LM |
| `prefix` | `str \| None` | `None` | Override field label in prompt |
| `**kwargs` | | | Passed to `pydantic.Field()` |

## dspy.OutputField

```python
dspy.OutputField(desc=None, prefix=None, type_=None, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desc` | `str \| None` | `None` | Natural-language description shown to LM |
| `prefix` | `str \| None` | `None` | Override field label in prompt |
| `type_` | `type \| None` | `None` | Alternative to Python type annotation |
| `**kwargs` | | | Passed to `pydantic.Field()` |

## Inline Signatures

String shorthand: `"input1, input2 -> output1, output2"`. Supports type suffixes: `str` (default), `int`, `float`, `bool`, `list[str]`.

```python
"question -> answer"
"text -> label: bool"
"context, question -> answer, confidence: float"
```
