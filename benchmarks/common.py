"""Completed-work timing shared by all benchmarks (including lazy MLX)."""

import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import tensorsmith as ts


def complete(device, values):
    # A queue barrier alone does NOT materialize lazy MLX expressions.
    ts.evaluate(values)
    ts.synchronize(device)


def measure(name, device, workload, iterations=20, warmup=3, units=None, completion=None):
    """workload returns every output/state update whose cost is being measured."""
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    completion = complete if completion is None else completion
    for _ in range(warmup):
        completion(device, workload())
    completion(device, [])
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        completion(device, workload())
        samples.append((time.perf_counter() - started) * 1000)
    result = {
        "name": name,
        "device": str(device),
        "iterations": iterations,
        "warmup": warmup,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
    }
    # Within-run sampling uncertainty only, not a confidence interval over
    # hardware, thermal state, seeds, or independently restarted executions.
    rng = np.random.default_rng(1729)
    bootstrap_medians = np.median(rng.choice(samples, (2000, len(samples)), replace=True), axis=1)
    result["median_bootstrap_95pct_interval_ms"] = np.percentile(
        bootstrap_medians, [2.5, 97.5]
    ).tolist()
    result["uncertainty_note"] = (
        "within-run bootstrap; correlated/thermal effects and between-run variation are not captured"
    )
    if units is not None:
        result["units_per_second"] = units / (result["median_ms"] / 1000)
    print(
        f"{device!s:8} {name:36} median={result['median_ms']:.3f}ms "
        f"p95={result['p95_ms']:.3f}ms"
        + (f" throughput={result['units_per_second']:.1f}/s" if units is not None else "")
    )
    return result


def report(results, args):
    versions = {}
    for name in ("numpy", "cupy-cuda12x", "mlx", "torch", "tensorsmith"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    hardware = {}
    if any("metal" in result["device"] for result in results):
        from tensorsmith.device import xp_for

        mx = xp_for("metal")
        if hasattr(mx, "device_info"):
            hardware["metal"] = mx.device_info()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
            hardware["metal"] = mx.metal.device_info()
    if any("cuda" in result["device"] for result in results):
        from tensorsmith.device import xp_for

        cp = xp_for("cuda")
        info = cp.cuda.runtime.getDeviceProperties(0)
        hardware["cuda"] = {
            "name": info["name"].decode(),
            "total_global_memory": info["totalGlobalMem"],
        }
    payload = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "torch_num_threads": sys.modules["torch"].get_num_threads()
            if "torch" in sys.modules
            else None,
            "hardware": hardware,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "versions": versions,
            "cpu_count": os.cpu_count(),
            "thread_environment": {
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "results": results,
        "methodology": "warmup excluded; wall-clock incl Python dispatch, explicit evaluation of "
        "returned outputs/gradients/optimizer state and device synchronization; "
        "not kernel-only timing; input allocation excluded except workload state",
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def add_timing_arguments(parser):
    parser.add_argument("--device", choices=["all", "cpu", "cuda", "metal"], default="cpu")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json", help="write machine-readable timings and environment metadata")


def devices(args):
    return ts.available_devices() if args.device == "all" else [ts.device(args.device)]
