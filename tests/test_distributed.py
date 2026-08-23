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
