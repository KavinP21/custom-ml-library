import unittest
from dataclasses import replace

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

    def test_checkpoint_dropout_replays_without_advancing_rng(self):
        for device in ts.available_devices():
            module = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.4), nn.GELU(), nn.Linear(8, 3)).to(
                device
            )
            raw = np.random.randn(2, 4).astype(np.float32)
            ts.seed(23)
            x = ts.tensor(raw, device=device, requires_grad=True)
            output = module(x)
            (output**2).sum().backward()
            expected = [x.grad.numpy()] + [p.grad.numpy() for p in module.parameters()]
            from tensorsmith.device import asnumpy, random_uniform

            expected_next = asnumpy(random_uniform((5,), device)).copy()
            module.zero_grad()
            x.zero_grad()
            ts.seed(23)
            checkpointed = nn.checkpoint(module, x)
            np.testing.assert_array_equal(checkpointed.numpy(), output.numpy())
            (checkpointed**2).sum().backward()
            actual = [x.grad.numpy()] + [p.grad.numpy() for p in module.parameters()]
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
            np.testing.assert_array_equal(asnumpy(random_uniform((5,), device)), expected_next)

    def test_numpy_random_interleaved_with_dropout_and_nested_checkpoint(self):
        x = ts.ones(8, requires_grad=True)

        def inner(input):
            return nn.functional.dropout(input, 0.5) * ts.rand(8)

        def function(input):
            return inner(input) + nn.checkpoint(inner, input * 2)

        ts.seed(42)
        expected = function(x)
        expected.sum().backward()
        derivative = x.grad.numpy().copy()
        x.zero_grad()
        ts.seed(42)
        actual = nn.checkpoint(function, x)
        actual.sum().backward()
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        np.testing.assert_array_equal(x.grad.numpy(), derivative)

    def test_checkpoint_lm_training_matches_regular(self):
        for device in ts.available_devices():
            config = nn.TransformerConfig(
                11,
                dim=16,
                num_heads=4,
                num_layers=2,
                dropout=0.2,
                max_seq_len=8,
                attention_backend="dense",
            )
            model = nn.TransformerLM(config, device=device)
            checkpointed = nn.TransformerLM(
                replace(config, activation_checkpointing=True), device=device
            )
            checkpointed.load_state_dict(model.state_dict())
            x, y = (
                ts.tensor([[1, 2, 3, 4]], device=device),
                ts.tensor([[2, 3, 4, 5]], device=device),
            )
            ts.seed(7)
            loss = model.loss(x, y)
            loss.backward()
            ts.seed(7)
            replay_loss = checkpointed.loss(x, y)
            replay_loss.backward()
            self.assertAlmostEqual(loss.item(), replay_loss.item(), places=5)
            for a, b in zip(model.parameters(), checkpointed.parameters()):
                np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), rtol=2e-4, atol=2e-5)

    def test_stateful_batchnorm_rejected_no_grad_is_passthrough(self):
        with self.assertRaises(ValueError):
            nn.checkpoint(nn.BatchNorm1d(4), ts.randn(2, 4, requires_grad=True))
        with ts.no_grad():
            output = nn.checkpoint(nn.Linear(4, 3), ts.randn(2, 4))
        self.assertFalse(output.requires_grad)

    def test_checkpoint_rejects_parameter_and_mode_changes(self):
        module = nn.Linear(3, 4)
        output = nn.checkpoint(module, ts.randn(2, 3))
        module.eval()
        with self.assertRaises(RuntimeError):
            output.sum().backward()
        module.train()
        output = nn.checkpoint(module, ts.randn(2, 3))
        module.weight._data = module.weight._data + 1
        with self.assertRaises(RuntimeError):
            output.sum().backward()


if __name__ == "__main__":
    unittest.main()
