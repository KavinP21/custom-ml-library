"""Prevent regressions back to timing graph construction on lazy backends."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


def load_common():
    location = Path(__file__).resolve().parents[1] / "benchmarks" / "common.py"
    spec = importlib.util.spec_from_file_location("benchmark_common", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkTests(unittest.TestCase):
    def test_evaluation_happens_before_device_barrier(self):
        common = load_common()
        events = []
        outputs = {"loss": object(), "gradients": [object()], "optimizer_state": [object()]}
        with (
            patch.object(
                common.ts, "evaluate", side_effect=lambda value: events.append(("eval", value))
            ),
            patch.object(
                common.ts, "synchronize", side_effect=lambda value: events.append(("sync", value))
            ),
            patch("builtins.print"),
        ):
            result = common.measure("test", "metal", lambda: outputs, iterations=3, warmup=2)
        self.assertEqual(len(result["samples_ms"]), 3)
        self.assertEqual(len(events), 12)
        for index in range(0, len(events), 2):
            self.assertEqual(events[index][0], "eval")
            self.assertEqual(events[index + 1], ("sync", "metal"))
        self.assertEqual(sum(event == ("eval", outputs) for event in events), 5)

    def test_invalid_counts_rejected(self):
        common = load_common()
        for iterations, warmup in ((0, 2), (-1, 2), (2, -1)):
            with self.assertRaises(ValueError):
                common.measure("test", "cpu", list, iterations, warmup)
