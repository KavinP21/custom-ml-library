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
    def test_large_half_mean_and_variance_accumulate_in_float32(self):
        for device in ts.available_devices():
            x = ts.ones(2, 32768, dtype="float16", device=device, requires_grad=True)
            result = x.mean()
            self.assertEqual(result.item(), 1.0)
            result.backward()
            np.testing.assert_array_equal(x.grad.numpy(), np.full(x.shape, 1 / 65536, np.float16))
            raw = np.tile(np.array([-1, 1], np.float16), 32768).reshape(2, -1)
            y = ts.tensor(raw, device=device, requires_grad=True)
            variance = y.var(correction=0)
            self.assertEqual(variance.item(), 1.0)
            variance.backward()
            np.testing.assert_array_equal(y.grad.numpy(), 2 * raw.astype(np.float32) / 65536)

    def test_numeric_scalars_reductions_and_gradients_preserve_precision(self):
        for device in ts.available_devices():
            for dtype in ("float32", "float16"):
                x = ts.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype, requires_grad=True)
                loss = ((x / 2 + 1.25) ** 2).mean()
                self.assertIn(dtype, str(loss.dtype))
                loss.backward()
                self.assertIn(dtype, str(x.grad.dtype))
                np.testing.assert_allclose(x.grad.numpy(), (x.numpy() / 2 + 1.25) / 3, rtol=3e-3)

    def test_mixed_precision_vjp_casts_to_parent_dtype(self):
        x = ts.tensor([1.0, 2.0], requires_grad=True, dtype="float32")
        y = ts.tensor([3.0, 4.0], requires_grad=True, dtype="float64")
        (x * y).sum().backward()
        self.assertEqual(x.grad.numpy().dtype, np.float32)
        self.assertEqual(y.grad.numpy().dtype, np.float64)
        np.testing.assert_array_equal(x.grad.numpy(), [3, 4])

    def test_branched_graph_accumulates(self):
        x = ts.tensor([1.0, -2.0, 3.0], requires_grad=True, dtype="float64")
        y = x * x
        loss = (y + y * 3).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), 8 * x.numpy())

    def test_broadcast_and_reduction_gradient(self):
        x = ts.randn(2, 3, 4, requires_grad=True, dtype="float64")
        bias = ts.randn(1, 3, 1, requires_grad=True, dtype="float64")
        ((x + bias) ** 2).mean().backward()
        expected = (2 * (x.numpy() + bias.numpy()) / x.size).sum(axis=(0, 2), keepdims=True)
        np.testing.assert_allclose(bias.grad.numpy(), expected, rtol=1e-10, atol=1e-10)

    def test_matmul_activation_finite_difference(self):
        np.random.seed(3)
        values = np.random.randn(2, 3)
        weights = np.random.randn(3, 2)
        x = ts.tensor(values, requires_grad=True, dtype="float64")
        w = ts.tensor(weights, requires_grad=True, dtype="float64")
        ((x @ w).tanh() ** 2).sum().backward()
        numerical = finite_difference(
            lambda v: np.square(np.tanh(v @ weights)).sum(), values.copy()
        )
        np.testing.assert_allclose(x.grad.numpy(), numerical, rtol=1e-6, atol=1e-6)

    def test_slice_scatter_and_repeated_indices(self):
        x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
        x[ts.tensor([0, 0, 2])].sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(), [2, 0, 1])

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

    def test_shape_tuple_and_cast_semantics(self):
        self.assertEqual(ts.zeros((2, 3)).shape, (2, 3))
        self.assertEqual(ts.randn((2, 3)).shape, (2, 3))
        x = ts.tensor([1.0, 2.0], requires_grad=True)
        self.assertFalse(x.long().requires_grad)

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


if __name__ == "__main__":
    unittest.main()
