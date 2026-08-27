"""Train linear regression with two independent CPU processes."""

import argparse
import multiprocessing as mp
import socket

import numpy as np

import tensorsmith as ts


def worker(rank, world_size, port, steps):
    ts.seed(42 + rank)
    with ts.distributed.TCPProcessGroup(rank, world_size, port=port) as group:
        model = ts.nn.Linear(2, 1)
        ddp = ts.distributed.DistributedDataParallel(model, group)
        optimizer = ts.optim.SGD(ddp.parameters(), lr=0.05)
        rng = np.random.default_rng(5)
        raw = rng.normal(size=(128, 2)).astype(np.float32)
        targets = raw @ np.array([[2.0], [-3.0]], np.float32) + 0.5
        dataset = ts.data.TensorDataset(ts.tensor(raw), ts.tensor(targets))
        sampler = ts.data.DistributedSampler(dataset, world_size, rank, shuffle=False)
        loader = ts.data.DataLoader(dataset, batch_size=32, sampler=sampler)
        for epoch in range(steps):
            for x, y in loader:
                optimizer.zero_grad()
                loss = ts.nn.MSELoss()(ddp(x), y)
                loss.backward()
                ddp.sync_gradients()
                optimizer.step()
            if rank == 0 and (epoch % 10 == 0 or epoch == steps - 1):
                print(f"epoch={epoch + 1} rank0_loss={loss.item():.6f}", flush=True)
        if rank == 0:
            print("weights:", model.weight.tolist(), "bias:", model.bias.tolist(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        port = handle.getsockname()[1]
    context = mp.get_context("spawn")
    workers = [
        context.Process(target=worker, args=(rank, 2, port, args.steps)) for rank in range(2)
    ]
    for process in workers:
        process.start()
    for process in workers:
        process.join()
    if any(process.exitcode != 0 for process in workers):
        raise RuntimeError("a distributed worker failed")


if __name__ == "__main__":
    main()
