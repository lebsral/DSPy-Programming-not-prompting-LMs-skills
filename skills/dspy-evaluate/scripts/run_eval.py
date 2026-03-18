"""Reusable evaluation harness for DSPy programs.

Usage:
    python scripts/run_eval.py --program path/to/program.json --devset path/to/devset.json --metric semantic_f1

Or import directly:
    from scripts.run_eval import run_eval
    results = run_eval(program, devset, metric_fn)
"""

import argparse
import json
from pathlib import Path

import dspy
from dspy.evaluate import Evaluate


def run_eval(
    program: dspy.Module,
    devset: list[dspy.Example],
    metric,
    num_threads: int = 4,
    display_progress: bool = True,
    display_table: int = 5,
) -> dict:
    """Run evaluation and return results summary.

    Args:
        program: Compiled DSPy program to evaluate.
        devset: List of DSPy Examples to evaluate against.
        metric: Metric function(example, prediction, trace=None) -> float.
        num_threads: Number of parallel threads for evaluation.
        display_progress: Show progress bar.
        display_table: Number of rows to show in results table (0 to hide).

    Returns:
        Dict with 'score', 'total', and 'results' keys.
    """
    evaluator = Evaluate(
        devset=devset,
        metric=metric,
        num_threads=num_threads,
        display_progress=display_progress,
        display_table=display_table,
    )

    score = evaluator(program)

    return {
        "score": score,
        "total": len(devset),
    }


# Built-in metrics for quick use
METRICS = {
    "semantic_f1": lambda: dspy.evaluate.SemanticF1(),
    "exact_match": lambda: dspy.evaluate.answer_exact_match,
}


def _load_devset(path: str) -> list[dspy.Example]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [dspy.Example(**row).with_inputs(*[k for k in row if k != "answer"]) for row in data]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a DSPy program")
    parser.add_argument("--program", required=True, help="Path to saved program (.json)")
    parser.add_argument("--devset", required=True, help="Path to dev set (.json)")
    parser.add_argument("--metric", default="semantic_f1", choices=list(METRICS.keys()))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    program = dspy.Module()
    program.load(args.program)
    devset = _load_devset(args.devset)
    metric = METRICS[args.metric]()

    results = run_eval(program, devset, metric, num_threads=args.threads)
    print(f"\nScore: {results['score']:.1f}% on {results['total']} examples")
