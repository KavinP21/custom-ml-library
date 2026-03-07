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

    def test_cross_entropy(self):
        logits = ts.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]], requires_grad=True)
        loss = F.cross_entropy(logits, ts.tensor([0, 2]))
        expected = -np.log([np.exp(2) / (np.exp(2) + 1 + np.exp(-1)), 1 / (np.exp(1) + 2)]).mean()
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        np.testing.assert_allclose(logits.grad.numpy().sum(1), 0, atol=1e-7)
