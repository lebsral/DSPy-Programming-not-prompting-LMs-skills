"""Monitoring setup template for DSPy programs in production.

Copy this file into your project and customize:
1. Configure your metrics backend (defaults to console logging)
2. Wrap your DSPy program with the monitor
3. Run your program as usual — metrics are collected automatically
"""

import json
import logging
import time
from dataclasses import dataclass, field
from functools import wraps

import dspy

logger = logging.getLogger(__name__)


@dataclass
class CallMetrics:
    """Metrics collected for each DSPy program call."""
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None
    token_usage: dict = field(default_factory=dict)


class DSPyMonitor:
    """Wraps a DSPy program to collect metrics on every call.

    Usage:
        program = dspy.ChainOfThought("question -> answer")
        monitored = DSPyMonitor(program, on_call=my_callback)
        result = monitored(question="What is 2+2?")
    """

    def __init__(self, program: dspy.Module, on_call=None):
        self.program = program
        self.on_call = on_call or self._default_log
        self.history: list[CallMetrics] = []

    def __call__(self, **kwargs):
        metrics = CallMetrics(input_keys=list(kwargs.keys()))
        start = time.time()

        try:
            result = self.program(**kwargs)
            metrics.output_keys = list(result.keys()) if hasattr(result, "keys") else []
            metrics.success = True
            return result
        except Exception as e:
            metrics.success = False
            metrics.error = str(e)
            raise
        finally:
            metrics.latency_ms = (time.time() - start) * 1000
            self.history.append(metrics)
            self.on_call(metrics)

    def _default_log(self, metrics: CallMetrics):
        status = "OK" if metrics.success else f"FAIL: {metrics.error}"
        logger.info(f"DSPy call: {status} in {metrics.latency_ms:.0f}ms")

    def summary(self) -> dict:
        """Return aggregate metrics."""
        if not self.history:
            return {"total_calls": 0}
        latencies = [m.latency_ms for m in self.history]
        return {
            "total_calls": len(self.history),
            "success_rate": sum(m.success for m in self.history) / len(self.history),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        }
