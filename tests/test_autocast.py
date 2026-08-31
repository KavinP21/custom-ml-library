import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import tensorsmith as ts


class AutocastTests(unittest.TestCase):
    def test_linear_casts_operators_but_keeps_parameters_and_gradients_fp32(self):
        ts.seed(13)
        model = ts.nn.Linear(4, 3)
        x = ts.randn(2, 4, requires_grad=True)
        expected = model(x).numpy()
        with ts.amp.autocast("cpu"):
            output = model(x)
            self.assertEqual(str(output.dtype), "float16")
        output.astype("float32").sum().backward()
        np.testing.assert_allclose(output.numpy(), expected, atol=2e-3, rtol=2e-3)
        for parameter in model.parameters():
            self.assertEqual(str(parameter.dtype), "float32")
            self.assertEqual(str(parameter.grad.dtype), "float32")
        self.assertEqual(str(x.grad.dtype), "float32")

    def test_nested_disable_reentry_and_exception_restore_policy(self):
        x = ts.ones(2, 2)
        context = ts.amp.autocast("cpu")
        with context:
            self.assertTrue(ts.amp.is_autocast_enabled("cpu"))
            with context:
                self.assertEqual(str((x @ x).dtype), "float16")
            with ts.amp.autocast("cpu", enabled=False):
                self.assertEqual(str((x @ x).dtype), "float32")
            self.assertEqual(str((x @ x).dtype), "float16")
        self.assertFalse(ts.amp.is_autocast_enabled("cpu"))
        with self.assertRaises(RuntimeError), ts.amp.autocast("cpu"):
            raise RuntimeError("body failed")
        self.assertFalse(ts.amp.is_autocast_enabled("cpu"))

    def test_float64_and_other_device_context_are_unchanged(self):
        x = ts.ones(2, 2, dtype="float64")
        with ts.amp.autocast("cpu"):
            self.assertEqual(str((x @ x).dtype), "float64")
        with ts.amp.autocast("cuda"):
            self.assertEqual(str((ts.ones(2, 2) @ ts.ones(2, 2)).dtype), "float32")
        with self.assertRaises(ValueError):
            ts.amp.autocast("cpu", dtype="bfloat16")

    def test_convolution_and_scaled_training_step(self):
        ts.seed(20)
        model = ts.nn.Conv2d(1, 2, 3)
        optimizer = ts.optim.AdamW(model.parameters())
        scaler = ts.amp.GradScaler(init_scale=128)
        before = model.weight.numpy().copy()
        with ts.amp.autocast("cpu"):
            output = model(ts.randn(2, 1, 5, 5))
            self.assertEqual(str(output.dtype), "float16")
            loss = (output.astype("float32") ** 2).mean()
        scaler.scale(loss).backward()
        self.assertTrue(scaler.step(optimizer))
        scaler.update()
        self.assertFalse(np.array_equal(before, model.weight.numpy()))

    def test_policy_is_thread_local(self):
        def worker(enabled):
            with ts.amp.autocast("cpu", enabled=enabled):
                return str((ts.ones(2, 2) @ ts.ones(2, 2)).dtype)

        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(worker, [True, False])), ["float16", "float32"])
