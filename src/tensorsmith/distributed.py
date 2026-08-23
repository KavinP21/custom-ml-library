"""Synchronous TCP collectives for small data-parallel training jobs.

Arrays use a length-prefixed, pickle-free protocol. Rank zero coordinates each
collective. This is a portable correctness baseline, not an NCCL replacement.
"""

from __future__ import annotations

import io
import json
import socket
import struct
import time

import numpy as np


class TCPProcessGroup:
    """All ranks must call the same collectives in the same order.

    Peers must be trusted. Use a private interface or loopback; this transport
    does not provide authentication or encryption. Timeouts fail a collective
    rather than hanging indefinitely. Each packet is limited to max_bytes.
    """

    def __init__(
        self,
        rank,
        world_size,
        *,
        host="127.0.0.1",
        port=29500,
        timeout=30.0,
        max_bytes=64 * 1024 * 1024,
    ):
        if (
            not isinstance(rank, int)
            or not isinstance(world_size, int)
            or not 0 <= rank < world_size
        ):
            raise ValueError("require 0 <= rank < world_size")
        if not 0 < port < 65536 or not np.isfinite(timeout) or timeout <= 0 or max_bytes <= 0:
            raise ValueError("invalid port, timeout or packet limit")
        self.rank, self.world_size = rank, world_size
        self.timeout, self.max_bytes = timeout, max_bytes
        self._sequence = 0
        self._closed = False
        self._listener = None
        self._connection = None
        self._peers = {}
        if world_size == 1:
            return
        deadline = time.monotonic() + timeout
        try:
            if rank == 0:
                self._listener = socket.socket()
                self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._listener.bind((host, port))
                self._listener.listen(world_size - 1)
                while len(self._peers) < world_size - 1:
                    self._listener.settimeout(max(0.001, deadline - time.monotonic()))
                    peer, _ = self._listener.accept()
                    self._configure(peer)
                    hello, _ = self._recv(peer)
                    peer_rank = hello.get("rank")
                    if (
                        hello.get("world_size") != world_size
                        or not isinstance(peer_rank, int)
                        or not 0 < peer_rank < world_size
                        or peer_rank in self._peers
                    ):
                        peer.close()
                        raise RuntimeError("invalid or duplicate process-group rank")
                    self._peers[peer_rank] = peer
                for peer in self._peers.values():
                    self._send(peer, {"ready": True})
            else:
                while True:
                    peer = socket.socket()
                    peer.settimeout(max(0.001, deadline - time.monotonic()))
                    try:
                        peer.connect((host, port))
                        break
                    except (ConnectionRefusedError, TimeoutError):
                        peer.close()
                        if time.monotonic() >= deadline:
                            raise TimeoutError("process-group connection timed out") from None
                        time.sleep(min(0.02, max(0, deadline - time.monotonic())))
                self._connection = peer
                self._configure(peer)
                self._send(peer, {"rank": rank, "world_size": world_size})
                ready, _ = self._recv(peer)
                if not ready.get("ready"):
                    raise RuntimeError("process-group initialization failed")
        except BaseException:
            self.close()
            raise

    def _configure(self, peer):
        peer.settimeout(self.timeout)
        peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _send(self, peer, header, value=None):
        metadata = json.dumps(header, separators=(",", ":")).encode()
        payload = b""
        if value is not None:
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            payload = buffer.getvalue()
        if len(metadata) + len(payload) > self.max_bytes:
            raise ValueError("collective exceeds the packet size limit")
        peer.sendall(struct.pack("!QQ", len(metadata), len(payload)) + metadata + payload)

    @staticmethod
    def _read_exact(peer, size):
        chunks = bytearray()
        while len(chunks) < size:
            chunk = peer.recv(size - len(chunks))
            if not chunk:
                raise ConnectionError("a process-group peer disconnected")
            chunks.extend(chunk)
        return bytes(chunks)

    def _recv(self, peer):
        nmeta, ndata = struct.unpack("!QQ", self._read_exact(peer, 16))
        if nmeta + ndata > self.max_bytes:
            raise ValueError("received packet exceeds the size limit")
        header = json.loads(self._read_exact(peer, nmeta))
        payload = self._read_exact(peer, ndata)
        value = np.load(io.BytesIO(payload), allow_pickle=False) if payload else None
        return header, value

    def _collective(self, kind, value, *, op="sum", src=0):
        if self._closed:
            raise RuntimeError("process group is closed")
        value = np.asarray(value)
        if value.dtype.kind not in "biufc":
            raise TypeError("collectives require numeric arrays")
        if op not in {"sum", "mean"} or not 0 <= src < self.world_size:
            raise ValueError("invalid reduction or source rank")
        if op == "mean" and value.dtype.kind not in "fc":
            raise TypeError("mean reduction requires floating or complex arrays")
        header = {
            "sequence": self._sequence,
            "kind": kind,
            "op": op,
            "src": src,
            "shape": list(value.shape),
            "dtype": value.dtype.str,
        }
        self._sequence += 1
        try:
            if self.rank != 0:
                self._send(self._connection, header, value)
                response, result = self._recv(self._connection)
                if "error" in response:
                    raise RuntimeError(response["error"])
                if response != header or result.shape != value.shape or result.dtype != value.dtype:
                    raise RuntimeError("invalid collective response")
                return result
            values = {0: value}
            errors = []
            for rank, peer in self._peers.items():
                request, incoming = self._recv(peer)
                if (
                    request != header
                    or incoming is None
                    or incoming.shape != value.shape
                    or incoming.dtype != value.dtype
                ):
                    errors.append(rank)
                values[rank] = incoming
            if errors:
                message = f"collective order/shape/dtype mismatch at ranks {errors}"
                for peer in self._peers.values():
                    self._send(peer, {"error": message})
                raise RuntimeError(message)
            if kind == "broadcast":
                result = values[src].copy()
            else:
                dtype = np.float32 if value.dtype == np.float16 else value.dtype
                result = np.zeros(value.shape, dtype=dtype)
                for rank in range(self.world_size):
                    result += values[rank].astype(dtype)
                if op == "mean":
                    result /= self.world_size
                result = result.astype(value.dtype)
            for peer in self._peers.values():
                self._send(peer, header, result)
            return result
        except BaseException:
            self.close()
            raise

    def all_reduce(self, value, *, op="sum"):
        return self._collective("all_reduce", value, op=op)

    def broadcast(self, value, *, src=0):
        return self._collective("broadcast", value, src=src)

    def barrier(self):
        self._collective("barrier", np.zeros(1, dtype=np.int64))

    def close(self):
        for peer in [self._connection, self._listener, *self._peers.values()]:
            if peer is not None:
                peer.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
