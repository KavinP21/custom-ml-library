import gc
import sys
import types
import unittest
import weakref
from unittest.mock import patch

import tensorsmith as ts


class DeviceTests(unittest.TestCase):
    def test_default_cuda_alias_is_device_zero(self):
        self.assertEqual(ts.device("cuda"), ts.device("cuda:0"))
        self.assertEqual(ts.Device("cuda").index, 0)
        self.assertNotEqual(ts.device("cuda:0"), ts.device("cuda:1"))

    def test_cpu_is_always_available(self):
        self.assertTrue(ts.is_available("cpu"))
        self.assertIn(ts.Device("cpu"), ts.available_devices())

    def test_unavailable_backend_has_actionable_error(self):
        for name in ("cuda", "metal"):
            if not ts.is_available(name):
                with self.assertRaisesRegex(RuntimeError, "pip install"):
                    ts.ones(2, device=name)

    def test_to_same_device_is_identity(self):
        value = ts.tensor([1.0])
        self.assertIs(value.to("cpu"), value)

    def test_evaluation_does_not_retain_outputs_until_cyclic_gc(self):
        class FakeMetalArray:
            pass

        FakeMetalArray.__module__ = "mlx.test"
        fake_core = types.ModuleType("mlx.core")
        fake_core.eval = lambda *arrays: None
        fake_package = types.ModuleType("mlx")
        fake_package.core = fake_core
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            with patch.dict(sys.modules, {"mlx": fake_package, "mlx.core": fake_core}):
                output = FakeMetalArray()
                reference = weakref.ref(output)
                ts.evaluate({"logits": [output]}, (None,))
                del output
                self.assertIsNone(reference(), "evaluation retained a completed GPU output")
        finally:
            if was_enabled:
                gc.enable()


if __name__ == "__main__":
    unittest.main()
