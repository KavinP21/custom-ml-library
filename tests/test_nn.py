import unittest

import numpy as np

import tensorsmith as ts
from tensorsmith import nn
from tensorsmith.nn import functional as F


def numerical_at(function, arrays, array_index, position, eps=1e-5):
    value = arrays[array_index][position]
    arrays[array_index][position] = value + eps
    high = function(*arrays)
    arrays[array_index][position] = value - eps
    low = function(*arrays)
    arrays[array_index][position] = value
    return (high - low) / (2 * eps)


class NeuralNetworkTests(unittest.TestCase):
    def test_half_batch_norm_large_statistics_stay_finite(self):
        raw = np.random.default_rng(52).normal(2, 3, (64, 3, 32, 32)).astype(np.float16)
        for device in ts.available_devices():
            layer = nn.BatchNorm2d(3).to(device, dtype="float16")
            x = ts.tensor(raw, device=device, requires_grad=True)
            output = layer(x)
            expected = (
                raw.astype(np.float32) - raw.astype(np.float32).mean((0, 2, 3), keepdims=True)
            ) / np.sqrt(raw.astype(np.float32).var((0, 2, 3), keepdims=True) + 1e-5)
            np.testing.assert_allclose(
                output.numpy(), expected.astype(np.float16), atol=3e-3, rtol=3e-3
            )
            (output**2).mean().backward()
            for value in (x.grad, layer.weight.grad, layer.bias.grad, layer.running_var):
                self.assertTrue(np.isfinite(value.numpy()).all())
            self.assertIn("float16", str(output.dtype))
            self.assertIn("float16", str(layer.running_var.dtype))

    def setUp(self):
        np.random.seed(5)

    def _check_operator(self, function, arrays, checks=6):
        values = [ts.tensor(a.copy(), requires_grad=True, dtype="float64") for a in arrays]
        function(*values).sum().backward()
        for array_index, original in enumerate(arrays):
            for flat in range(min(checks, original.size)):
                position = np.unravel_index(flat, original.shape)
                numerical = numerical_at(
                    lambda *raw: (
                        function(*[ts.tensor(x, dtype="float64") for x in raw]).sum().item()
                    ),
                    arrays,
                    array_index,
                    position,
                )
                self.assertAlmostEqual(
                    values[array_index].grad.numpy()[position], numerical, places=5
                )

    def test_conv1d_backward(self):
        x = np.random.randn(1, 2, 5)
        weight = np.random.randn(3, 2, 3)
        self._check_operator(lambda a, b: F.conv1d(a, b, padding=1), [x, weight])

    def test_grouped_strided_conv2d_backward(self):
        x = np.random.randn(1, 2, 4, 5)
        weight = np.random.randn(4, 1, 2, 2)
        self._check_operator(
            lambda a, b: F.conv2d(a, b, stride=(2, 1), padding=(1, 0), groups=2),
            [x, weight],
        )

    def test_pooling_values_and_gradients(self):
        raw = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        x = ts.tensor(raw, requires_grad=True)
        maximum = F.max_pool2d(x, 2, 2)
        np.testing.assert_array_equal(maximum.numpy(), [[[[5, 7], [13, 15]]]])
        maximum.sum().backward()
        expected = np.zeros_like(raw)
        expected[:, :, 1::2, 1::2] = 1
        np.testing.assert_array_equal(x.grad.numpy(), expected)
        self._check_operator(lambda a: F.avg_pool2d(a, 2, stride=1), [np.random.randn(1, 1, 4, 4)])

    def test_max_pool_ties_route_to_first_winner(self):
        for device in ts.available_devices():
            x = ts.ones(1, 1, 2, 2, device=device, requires_grad=True)
            F.max_pool2d(x, 2).sum().backward()
            np.testing.assert_array_equal(x.grad.numpy(), [[[[1, 0], [0, 0]]]])

    def test_embedding_repeated_index_gradient(self):
        layer = nn.Embedding(5, 3)
        layer(ts.tensor([1, 1, 3])).sum().backward()
        expected = np.zeros((5, 3), dtype=np.float32)
        expected[1] = 2
        expected[3] = 1
        np.testing.assert_array_equal(layer.weight.grad.numpy(), expected)

    def test_sequential_registration_and_state(self):
        model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
        self.assertEqual(len(list(model.parameters())), 4)
        self.assertEqual(
            set(model.state_dict()),
            {"layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias"},
        )
        clone = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
        clone.load_state_dict(model.state_dict())
        x = ts.randn(4, 3)
        np.testing.assert_allclose(model(x).numpy(), clone(x).numpy())

    def test_layer_norm(self):
        x = ts.randn(3, 4, 5, requires_grad=True)
        output = nn.LayerNorm(5)(x)
        np.testing.assert_allclose(output.numpy().mean(-1), 0, atol=2e-6)
        np.testing.assert_allclose(output.numpy().var(-1), 1, atol=2e-4)
        output.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)

    def test_batch_norm_tracks_and_uses_running_stats(self):
        layer = nn.BatchNorm2d(3, momentum=1.0)
        x = ts.tensor(np.random.randn(4, 3, 2, 2).astype(np.float32), requires_grad=True)
        training_output = layer(x)
        np.testing.assert_allclose(training_output.numpy().mean((0, 2, 3)), 0, atol=2e-6)
        training_output.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)
        saved_mean = layer.running_mean.numpy()
        layer.eval()
        evaluation_output = layer(x.detach())
        expected = (x.numpy() - saved_mean.reshape(1, 3, 1, 1)) / np.sqrt(
            layer.running_var.numpy().reshape(1, 3, 1, 1) + layer.eps
        )
        np.testing.assert_allclose(evaluation_output.numpy(), expected, atol=2e-6)

    def test_cross_entropy(self):
        logits = ts.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]], requires_grad=True)
        loss = F.cross_entropy(logits, ts.tensor([0, 2]))
        expected = -np.log([np.exp(2) / (np.exp(2) + 1 + np.exp(-1)), 1 / (np.exp(1) + 2)]).mean()
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        np.testing.assert_allclose(logits.grad.numpy().sum(1), 0, atol=1e-7)

    def test_batch_norm_running_variance_is_unbiased(self):
        raw = np.arange(24, dtype=np.float32).reshape(3, 2, 2, 2)
        for device in ts.available_devices():
            layer = nn.BatchNorm2d(2, momentum=1.0, device=device)
            layer(ts.tensor(raw, device=device))
            np.testing.assert_allclose(
                layer.running_var.numpy(), raw.var((0, 2, 3), ddof=1), rtol=2e-6
            )
            with self.assertRaisesRegex(ValueError, "more than one"):
                layer(ts.ones(1, 2, 1, 1, device=device))


if __name__ == "__main__":
    unittest.main()
