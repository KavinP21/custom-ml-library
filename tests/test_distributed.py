import multiprocessing as mp
import queue
import socket
import unittest

import numpy as np

import tensorsmith as ts


def _collective_worker(rank, port, mode, output):
    try:
        with ts.distributed.TCPProcessGroup(rank, 2, port=port, timeout=5) as group:
            if mode == "mismatch":
                group.all_reduce(np.ones(rank + 1))
            else:
                reduced = group.all_reduce(np.array([rank + 1, 2 * rank], np.float32), op="mean")
                broadcast = group.broadcast(np.array([rank], np.int64), src=1)
                group.barrier()
                output.put((rank, "ok", reduced.tolist(), broadcast.tolist()))
                return
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        output.put((rank, "error", str(error)))


def _port():
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def _training_worker(rank, port, mode, output):
    try:
        with ts.distributed.TCPProcessGroup(rank, 2, port=port, timeout=5) as group:
            ts.seed(42 + rank)
            model = ts.nn.Linear(2 if mode != "schema" else rank + 1, 1)
            wrapped = ts.distributed.DistributedDataParallel(model, group)
            if mode == "schema":
                raise RuntimeError("schema mismatch was not detected")
            optimizer = ts.optim.SGD(wrapped.parameters(), lr=0.1, momentum=0.9)
            raw = np.arange(12, dtype=np.float32).reshape(6, 2) / 8
            targets = raw @ np.array([[0.5], [-0.25]], np.float32) + 0.3
            local = slice(rank * 3, (rank + 1) * 3)
            for _ in range(5):
                optimizer.zero_grad()
                loss = ts.nn.MSELoss()(wrapped(ts.tensor(raw[local])), ts.tensor(targets[local]))
                loss.backward()
                wrapped.sync_gradients()
                optimizer.step()
            output.put((rank, "ok", model.weight.numpy().tolist(), model.bias.numpy().tolist()))
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        output.put((rank, "error", str(error)))


def _run_workers(target, mode):
    context = mp.get_context("spawn")
    output = context.Queue()
    port = _port()
    workers = [context.Process(target=target, args=(rank, port, mode, output)) for rank in range(2)]
    for worker in workers:
        worker.start()
    try:
        results = [output.get(timeout=15) for _ in workers]
        for worker in workers:
            worker.join(timeout=5)
            if worker.exitcode != 0:
                raise RuntimeError(f"worker failed with exit code {worker.exitcode}")
        return sorted(results)
    except queue.Empty as error:
        raise RuntimeError("distributed workers did not finish") from error
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join()
        output.close()


class CollectiveTests(unittest.TestCase):
    def test_distributed_updates_match_global_batch_training(self):
        rows = _run_workers(_training_worker, "train")
        ts.seed(42)
        model = ts.nn.Linear(2, 1)
        optimizer = ts.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        raw = np.arange(12, dtype=np.float32).reshape(6, 2) / 8
        targets = raw @ np.array([[0.5], [-0.25]], np.float32) + 0.3
        for _ in range(5):
            optimizer.zero_grad()
            loss = ts.nn.MSELoss()(model(ts.tensor(raw)), ts.tensor(targets))
            loss.backward()
            optimizer.step()
        for row in rows:
            self.assertEqual(row[1], "ok", row)
            np.testing.assert_allclose(row[2], model.weight.numpy(), atol=2e-7, rtol=2e-6)
            np.testing.assert_allclose(row[3], model.bias.numpy(), atol=2e-7, rtol=2e-6)

    def test_model_schema_mismatch_is_reported_to_all_ranks(self):
        rows = _run_workers(_training_worker, "schema")
        for row in rows:
            self.assertEqual(row[1], "error", row)
            self.assertIn("schemas differ", row[2])

    def test_sampler_epoch_padding_and_loader(self):
        dataset = ts.data.TensorDataset(ts.arange(7))
        samplers = [
            ts.data.DistributedSampler(dataset, 2, rank, shuffle=False) for rank in range(2)
        ]
        self.assertEqual(list(samplers[0]), [0, 2, 4, 6])
        self.assertEqual(list(samplers[1]), [1, 3, 5, 0])
        loader = ts.data.DataLoader(dataset, batch_size=3, sampler=samplers[1])
        self.assertEqual(len(loader), 2)
        self.assertEqual([batch[0].tolist() for batch in loader], [[1, 3, 5], [0]])
        a = ts.data.DistributedSampler(list(range(20)), 2, 0, seed=12)
        b = ts.data.DistributedSampler(list(range(20)), 2, 0, seed=12)
        self.assertEqual(list(a), list(b))
        original = list(a)
        a.set_epoch(1)
        self.assertNotEqual(original, list(a))
        self.assertEqual(list(ts.data.DistributedSampler([], 3, 1)), [])
        self.assertEqual(len(ts.data.DistributedSampler(list(range(2)), 3, 0, drop_last=True)), 0)

    def test_real_two_process_reduce_broadcast_and_barrier(self):
        rows = _run_workers(_collective_worker, "normal")
        for row in rows:
            self.assertEqual(row[1], "ok", row)
            np.testing.assert_array_equal(row[2], [1.5, 1])
            self.assertEqual(row[3], [1])

    def test_mismatch_reaches_both_ranks(self):
        rows = _run_workers(_collective_worker, "mismatch")
        for row in rows:
            self.assertEqual(row[1], "error", row)
            self.assertIn("mismatch", row[2])

    def test_missing_rank_times_out_and_closes(self):
        with self.assertRaises(TimeoutError):
            ts.distributed.TCPProcessGroup(0, 2, port=_port(), timeout=0.1)

    def test_single_rank_and_invalid_dtypes(self):
        with ts.distributed.TCPProcessGroup(0, 1) as group:
            np.testing.assert_array_equal(group.all_reduce(np.array([1.0])), [1])
            with self.assertRaises(TypeError):
                group.all_reduce(np.array(["unsafe"]))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            group.barrier()
