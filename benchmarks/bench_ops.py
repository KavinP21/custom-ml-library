"""Matmul forward/backward latency, explicitly materializing lazy gradients."""

import argparse

from common import add_timing_arguments, devices, measure, report
from reference import torch_backend

import tensorsmith as ts


def benchmark(device: str, size: int, iterations: int, warmup: int = 3, torch_compare=False):
    ts.seed(42)
    if size <= 0:
        raise ValueError("size must be positive")
    x = ts.randn(size, size, device=device, requires_grad=True)
    weight = ts.randn(size, size, device=device, requires_grad=True)
    from common import complete

    complete(device, (x, weight))

    def workload():
        x.zero_grad()
        weight.zero_grad()
        loss = (x @ weight).mean()
        loss.backward()
        return loss, x.grad, weight.grad

    results = [measure(f"matmul_{size}_forward_backward", device, workload, iterations, warmup)]
    if torch_compare:
        torch, selected, torch_complete = torch_backend(ts.device(device))
        tx, tw = [
            torch.tensor(t.numpy().copy(), device=selected, requires_grad=True) for t in (x, weight)
        ]

        def reference():
            tx.grad = tw.grad = None
            loss = (tx @ tw).mean()
            loss.backward()
            return loss, tx.grad, tw.grad

        import numpy as np

        np.testing.assert_allclose(
            reference()[0].detach().cpu().numpy(), workload()[0].numpy(), rtol=1e-4, atol=1e-5
        )
        for ours, theirs in ((x, tx), (weight, tw)):
            np.testing.assert_allclose(
                ours.grad.numpy(), theirs.grad.detach().cpu().numpy(), rtol=1e-4, atol=1e-5
            )
        results.append(
            measure(
                f"torch_matmul_{size}_forward_backward",
                device,
                reference,
                iterations,
                warmup,
                completion=torch_complete,
            )
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_timing_arguments(parser)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--torch",
        action="store_true",
        help="compare to optional PyTorch on the same hardware/dtype",
    )
    args = parser.parse_args()
    report(
        [
            r
            for d in devices(args)
            for r in benchmark(str(d), args.size, args.iterations, args.warmup, args.torch)
        ],
        args,
    )
