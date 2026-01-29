import unittest


import numpy as np


import tensorsmith as ts


def finite_difference(function, value, eps=1e-5):
    result = np.zeros_like(value, dtype=np.float64)
    for index in np.ndindex(value.shape):
        old = value[index]
        value[index] = old + eps
        high = function(value.copy())
        value[index] = old - eps
        low = function(value.copy())
        value[index] = old
        result[index] = (high - low) / (2 * eps)
    return result


class AutogradTests(unittest.TestCase):
    def test_branched_graph_accumulates(self):
        x = ts.tensor([1.0, -2.0, 3.0], requires_grad=True, dtype="float64")
        y = x * x
        loss = (y + y * 3).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), 8 * x.numpy())

    def test_non_scalar_needs_gradient(self):
        value = ts.ones(2, requires_grad=True)
        with self.assertRaises(RuntimeError):
            value.backward()
        value.backward(ts.tensor([2.0, 3.0]))
        np.testing.assert_array_equal(value.grad.numpy(), [2, 3])

    def test_no_grad(self):
        x = ts.ones(2, requires_grad=True)
        with ts.no_grad():
            y = x * 4
        self.assertFalse(y.requires_grad)

    def test_gradient_accumulates_across_passes(self):
        x = ts.tensor(2.0, requires_grad=True)
        (x * 3).backward()
        (x * 4).backward()
        self.assertEqual(x.grad.item(), 7)

    def test_tape_lifetime_is_explicit(self):
        x = ts.tensor(2.0, requires_grad=True)
        loss = x * x
        loss.backward()
        with self.assertRaisesRegex(RuntimeError, "retain_graph"):
            loss.backward()

        retained = x * 3
        retained.backward(retain_graph=True)
        retained.backward()
        self.assertEqual(x.grad.item(), 4 + 3 + 3)
