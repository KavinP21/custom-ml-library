import unittest

import numpy as np

import tensorsmith as ts

CUDA = ts.is_available("cuda")


class CUDAGuardTests(unittest.TestCase):
    def test_capture_rejects_cpu_and_uncaptured_replay(self):
        with self.assertRaisesRegex(ValueError, "CUDA tensors"):
            ts.cuda.CUDAGraph().capture(ts.nn.Linear(2, 1).eval(), ts.ones(1, 2))
        with self.assertRaisesRegex(RuntimeError, "before replay"):
            ts.cuda.CUDAGraph().replay()


@unittest.skipUnless(CUDA, "requires an actual NVIDIA CUDA device")
class CUDAExecutionTests(unittest.TestCase):
    def test_graph_replay_matches_eager_and_reuses_storage(self):
        ts.seed(7)
        model = ts.nn.Linear(4, 3, device="cuda").eval()
        x = ts.randn(2, 4, device="cuda")
        graph = ts.cuda.CUDAGraph().capture(model, x)
        for factor in (0.5, 2.0, -1.0):
            incoming = x * factor
            with ts.no_grad():
                expected = model(incoming)
            actual = graph.replay(incoming)
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=2e-6, rtol=2e-6)
            self.assertIs(actual, graph.output)
        with self.assertRaises(ValueError):
            graph.replay(ts.ones(1, 4, device="cuda"))
        model.weight._data = model.weight._data.copy()
        with self.assertRaisesRegex(RuntimeError, "storage"):
            graph.replay(x)

    def test_autocast_backward_on_cuda(self):
        model = ts.nn.Linear(4, 3, device="cuda")
        with ts.amp.autocast("cuda"):
            loss = model(ts.ones(2, 4, device="cuda")).astype("float32").mean()
        loss.backward()
        for parameter in model.parameters():
            self.assertEqual(str(parameter.dtype), "float32")
            self.assertTrue(np.isfinite(parameter.grad.numpy()).all())
