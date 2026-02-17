import unittest


import numpy as np


import tensorsmith as ts


from tensorsmith import nn


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
