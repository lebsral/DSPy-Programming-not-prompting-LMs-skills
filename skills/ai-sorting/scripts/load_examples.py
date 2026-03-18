"""Load labeled examples from CSV or JSON into DSPy Examples with train/dev split.

Usage (from SKILL.md or Claude):
    from scripts.load_examples import load_examples
    trainset, devset = load_examples("data.csv", input_keys=["text"], label_key="category")
"""

import json
import random
from pathlib import Path

import dspy


def load_examples(
    path: str,
    input_keys: list[str],
    label_key: str = "label",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load labeled data and split into train/dev sets.

    Supports CSV (.csv) and JSON/JSONL (.json, .jsonl) files.

    Args:
        path: Path to data file.
        input_keys: Column/field names to use as inputs.
        label_key: Column/field name for the label.
        train_ratio: Fraction of data for training (rest goes to dev).
        seed: Random seed for reproducible splits.

    Returns:
        (trainset, devset) tuple of DSPy Example lists.
    """
    path = Path(path)
    rows = _load_rows(path)

    examples = []
    for row in rows:
        fields = {k: row[k] for k in input_keys}
        fields[label_key] = row[label_key]
        ex = dspy.Example(**fields).with_inputs(*input_keys)
        examples.append(ex)

    random.seed(seed)
    random.shuffle(examples)
    split = int(len(examples) * train_ratio)
    return examples[:split], examples[split:]


def _load_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import csv

        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    elif suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            raise ValueError("JSON file must contain a top-level array of objects")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv, .json, or .jsonl")
