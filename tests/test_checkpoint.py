import unittest


import numpy as np


import tensorsmith as ts


from tensorsmith import nn


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(31)

    def test_grad_does_not_touch_buffers_and_supports_upstream(self):
        for device in ts.available_devices():
            x = ts.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
            x.grad = ts.tensor([5.0, 6.0, 7.0], device=device)
            output = x * x + x
            (derivative,) = ts.grad(
                output, x, ts.tensor([2.0, 3.0, 4.0], device=device), retain_graph=True
            )
            np.testing.assert_allclose(derivative.numpy(), [6, 15, 28])
            np.testing.assert_array_equal(x.grad.numpy(), [5, 6, 7])
            self.assertIsNone(output.grad)
            output.sum().backward()
            np.testing.assert_array_equal(x.grad.numpy(), [8, 11, 14])
            self.assertFalse(derivative.requires_grad)

    def test_grad_unused_and_freed_tape_errors(self):
        x, unused = ts.randn(3, requires_grad=True), ts.randn(3, requires_grad=True)
        output = (x * x).sum()
        gradients = ts.grad(output, (x, unused), allow_unused=True)
        self.assertIsNone(gradients[1])
        with self.assertRaises(RuntimeError):
            ts.grad(output, x)
        y = x * 3
        y.sum().backward()
        with self.assertRaises(RuntimeError):
            (y * 2).sum().backward()

    def test_checkpoint_module_and_captured_parameter_vjp(self):
        for device in ts.available_devices():
            module = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3)).to(device)
            x = ts.randn(2, 4, device=device, requires_grad=True)
            expected = ts.grad((module(x) ** 2).sum(), (x, *module.parameters()))
            calls = []

            def function(input, calls=calls, module=module):
                calls.append(ts.is_grad_enabled())
                return module(input)

            output = nn.checkpoint(function, x, parameters=module.parameters())
            self.assertEqual(output._op, "checkpoint")
            self.assertEqual(calls, [False])
            actual = ts.grad((output**2).sum(), (x, *module.parameters()))
            self.assertEqual(calls, [False, True])
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-5)
            self.assertTrue(all(p.grad is None for p in module.parameters()))
            nn.checkpoint(module, x).sum().backward()
            self.assertTrue(all(p.grad is not None for p in module.parameters()))
