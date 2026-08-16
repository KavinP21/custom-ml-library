import unittest

import numpy as np

import tensorsmith as ts


class Square(ts.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, upstream):
        (x,) = ctx.saved_tensors
        return 2 * x * upstream


class CustomAutogradTests(unittest.TestCase):
    def test_custom_backward_composes_with_branched_graph(self):
        x = ts.tensor([1.5, -2.0], requires_grad=True, dtype="float64")
        output = Square.apply(x)
        (output * x + output).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), 3 * x.numpy() ** 2 + 2 * x.numpy())

    def test_saved_values_are_snapshots(self):
        x = ts.tensor([2.0], requires_grad=True)
        output = Square.apply(x)
        x._data[...] = 10
        output.backward()
        np.testing.assert_array_equal(x.grad.numpy(), [4.0])

    def test_numerical_direction_and_no_grad(self):
        raw = np.array([-0.5, 1.2, 2.5])
        direction = np.array([0.1, 0.7, -0.4])
        x = ts.tensor(raw, requires_grad=True)
        (derivative,) = ts.grad(Square.apply(x).tanh().sum(), x)

        def f(value):
            return Square.apply(ts.tensor(value)).tanh().sum().item()

        numerical = (f(raw + 1e-5 * direction) - f(raw - 1e-5 * direction)) / 2e-5
        self.assertAlmostEqual((derivative.numpy() * direction).sum(), numerical, places=7)
        with ts.no_grad():
            self.assertFalse(Square.apply(x).requires_grad)

    def test_gradient_contract_and_unused_input(self):
        class FirstOnly(ts.autograd.Function):
            @staticmethod
            def forward(ctx, x, y, *, factor):
                ctx.factor = factor
                return x * factor

            @staticmethod
            def backward(ctx, upstream):
                return upstream * ctx.factor, None

        x, y = ts.ones(2, requires_grad=True), ts.ones(2, requires_grad=True)
        FirstOnly.apply(x, y, factor=3).sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(), [3, 3])
        self.assertIsNone(y.grad)

        class WrongShape(Square):
            @staticmethod
            def backward(ctx, upstream):
                return ts.ones(3)

        with self.assertRaisesRegex(ValueError, "shape/device"):
            WrongShape.apply(ts.ones(2, requires_grad=True)).sum().backward()
