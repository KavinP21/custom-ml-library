"""Run on CPU and every accelerator visible to the current process."""

import unittest

import numpy as np

import tensorsmith as ts
from tensorsmith.nn import functional as F


class BackendParityTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(11)

    def test_compound_autograd_parity(self):
        raw_x = np.random.randn(2, 3).astype(np.float32)
        raw_w = np.random.randn(3, 4).astype(np.float32)
        expected = None
        for selected in ts.available_devices():
            with self.subTest(device=str(selected)):
                np.testing.assert_array_equal(ts.ones(2, device=selected).numpy(), [1, 1])
                np.testing.assert_array_equal(ts.zeros(2, device=selected).numpy(), [0, 0])
                np.testing.assert_array_equal(ts.arange(3, device=selected).numpy(), [0, 1, 2])
                x = ts.tensor(raw_x, device=selected, requires_grad=True)
                weight = ts.tensor(raw_w, device=selected, requires_grad=True)
                loss = ((x @ weight).tanh() ** 2).mean()
                loss.backward()
                current = (loss.item(), x.grad.numpy(), weight.grad.numpy())
                if expected is None:
                    expected = current
                else:
                    self.assertAlmostEqual(current[0], expected[0], places=5)
                    np.testing.assert_allclose(current[1], expected[1], rtol=2e-5, atol=2e-5)
                    np.testing.assert_allclose(current[2], expected[2], rtol=2e-5, atol=2e-5)

    def test_conv_pool_embedding_and_optimizer(self):
        raw_x = np.random.randn(1, 2, 5, 5).astype(np.float32)
        raw_w = np.random.randn(4, 1, 3, 2).astype(np.float32)
        expected = None
        for selected in ts.available_devices():
            with self.subTest(device=str(selected)):
                x = ts.tensor(raw_x, device=selected, requires_grad=True)
                weight = ts.tensor(raw_w, device=selected, requires_grad=True)
                output = F.conv2d(x, weight, stride=(2, 1), padding=(1, 0), groups=2)
                pooled = F.avg_pool2d(output, 2, stride=1)
                pooled.sum().backward()
                current = (pooled.numpy(), x.grad.numpy(), weight.grad.numpy())
                if expected is None:
                    expected = current
                else:
                    for got, wanted in zip(current, expected):
                        np.testing.assert_allclose(got, wanted, rtol=3e-5, atol=3e-5)

                embedding = ts.nn.Embedding(6, 3, device=selected)
                embedding(ts.tensor([1, 1, 4], device=selected)).sum().backward()
                self.assertEqual(embedding.weight.grad.numpy()[1].tolist(), [2.0, 2.0, 2.0])

                optimizer = ts.optim.AdamW(embedding.parameters(), lr=0.01)
                before = embedding.weight.numpy()
                optimizer.step()
                self.assertFalse(np.array_equal(before, embedding.weight.numpy()))
                norm = ts.nn.clip_grad_norm_(embedding.parameters(), 1.0)
                self.assertGreater(norm, 0)
                ts.synchronize(selected)


if __name__ == "__main__":
    unittest.main()
