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

    def test_kernel_requires_cuda_and_missing_runtime_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            ts.cuda.RawKernel("source", "name", device_name="cpu")
        if not CUDA:
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                ts.cuda.RawKernel('extern "C" __global__ void noop() {}', "noop").compile()


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

    def test_raw_kernel_with_custom_backward(self):
        kernel = ts.cuda.RawKernel(
            'extern "C" __global__ void square(const float* x, float* y, int n) { int i = blockDim.x * blockIdx.x + threadIdx.x; if (i < n) y[i] = x[i] * x[i]; }',
            "square",
        )

        class Square(ts.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                output = ts.empty(*x.shape, device=x.device)
                kernel(((x.size + 127) // 128,), (128,), (x, output, np.int32(x.size)))
                return output

            @staticmethod
            def backward(ctx, upstream):
                return 2 * ctx.saved_tensors[0] * upstream

        x = ts.tensor([1.0, -2.0, 3.0], device="cuda", requires_grad=True)
        y = Square.apply(x)
        y.sum().backward()
        np.testing.assert_array_equal(y.numpy(), [1, 4, 9])
        np.testing.assert_array_equal(x.grad.numpy(), [2, -4, 6])

    def test_autocast_backward_on_cuda(self):
        model = ts.nn.Linear(4, 3, device="cuda")
        with ts.amp.autocast("cuda"):
            loss = model(ts.ones(2, 4, device="cuda")).astype("float32").mean()
        loss.backward()
        for parameter in model.parameters():
            self.assertEqual(str(parameter.dtype), "float32")
            self.assertTrue(np.isfinite(parameter.grad.numpy()).all())
