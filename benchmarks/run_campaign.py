"""Serial local validation campaign; logs and failures survive long executions.

The two full-data TensorSmith training runs must already be complete (or the
explicit --wait-for report can be running). No cloud compute or concurrent
GPU jobs are created. Large-model capped runs are labeled STRESS, not quality.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(args):
    root = Path(__file__).resolve().parents[1]
    results = root / "benchmarks/results"
    checkpoints = root / "benchmarks/checkpoints"
    log_dir = results / "campaign-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.wait_for:
        while json.loads(args.wait_for.read_text())["status"] != "complete":
            time.sleep(10)
    language = checkpoints / "wikitext2-tensorsmith-metal-seed42.npz"
    vision = checkpoints / "cifar10-tensorsmith-metal-seed42.npz"
    large = checkpoints / "wikitext2-129m-stress-metal.npz"
    jobs = []

    def job(name, script, options):
        jobs.append(
            (name, [sys.executable, str(root / script), *[str(value) for value in options]])
        )

    job(
        "torch-wikitext2-full",
        "benchmarks/bench_tasks.py",
        [
            "--task",
            "wikitext2",
            "--engine",
            "torch",
            "--device",
            "metal",
            "--epochs",
            10,
            "--batch-size",
            16,
            "--eval-batch-size",
            16,
            "--checkpoint",
            checkpoints / "wikitext2-torch-metal-seed42.npz",
            "--output",
            results / "wikitext2-torch-metal-seed42.json",
        ],
    )
    job(
        "torch-cifar10-full",
        "benchmarks/bench_tasks.py",
        [
            "--task",
            "cifar10",
            "--engine",
            "torch",
            "--device",
            "metal",
            "--epochs",
            30,
            "--batch-size",
            64,
            "--eval-batch-size",
            64,
            "--lr",
            0.1,
            "--min-lr",
            0.001,
            "--weight-decay",
            0.0005,
            "--warmup-steps",
            100,
            "--checkpoint",
            checkpoints / "cifar10-torch-metal-seed42.npz",
            "--output",
            results / "cifar10-torch-metal-seed42.json",
        ],
    )
    job(
        "129m-real-text-STRESS",
        "benchmarks/bench_tasks.py",
        [
            "--task",
            "wikitext2",
            "--engine",
            "tensorsmith",
            "--device",
            "metal",
            "--epochs",
            1,
            "--max-steps",
            5,
            "--timing-warmup",
            1,
            "--batch-size",
            2,
            "--eval-batch-size",
            4,
            "--seq-len",
            512,
            "--dim",
            768,
            "--layers",
            12,
            "--heads",
            12,
            "--kv-heads",
            4,
            "--hidden-dim",
            3072,
            "--warmup-steps",
            0,
            "--lr",
            0.0001,
            "--min-lr",
            0.00001,
            "--checkpoint",
            large,
            "--output",
            results / "wikitext2-129m-STRESS-metal.json",
        ],
    )
    if args.refresh_tensorsmith:
        # Repeat FULL task training after a runtime fix, reusing the completed
        # matched reference runs. Call with --start-at tensorsmith-wikitext2-refresh
        # to preserve reference/stress evidence without duplicating that work.
        for task, (_, command) in zip(("wikitext2", "cifar10"), jobs[:2]):
            job(
                f"tensorsmith-{task}-refresh",
                "benchmarks/bench_tasks.py",
                [value.replace("torch", "tensorsmith") for value in command[2:]],
            )
        job(
            "129m-real-text-STRESS-refresh",
            "benchmarks/bench_tasks.py",
            jobs[2][1][2:],
        )
    job(
        "metal-acceptance",
        "benchmarks/verify_tasks.py",
        [
            "--device",
            "metal",
            "--long-context",
            "--language-checkpoint",
            language,
            "--vision-checkpoint",
            vision,
            "--large-checkpoint",
            large,
            "--output",
            results / "real-task-acceptance-metal.json",
        ],
    )
    jobs.append(("metal-unit-suite", [sys.executable, "-m", "pytest", "-ra"]))
    for repeat in range(1, 4):
        for task, checkpoint, batch in [("wikitext2", language, 16), ("cifar10", vision, 64)]:
            # Alternate engine order across fresh-process replicates.
            engines = ["tensorsmith", "torch"] if repeat % 2 else ["torch", "tensorsmith"]
            for engine in engines:
                job(
                    f"isolated-{task}-{engine}-r{repeat}",
                    "benchmarks/bench_task_speed.py",
                    [
                        "--engine",
                        engine,
                        "--device",
                        "metal",
                        "--batch-size",
                        batch,
                        "--checkpoint",
                        checkpoint,
                        "--iterations",
                        50,
                        "--warmup",
                        10,
                        "--output",
                        results / f"{task}-isolated-{engine}-metal-r{repeat}.json",
                    ],
                )
    job(
        "cpu-acceptance",
        "benchmarks/verify_tasks.py",
        [
            "--device",
            "cpu",
            "--long-context",
            "--language-checkpoint",
            language,
            "--vision-checkpoint",
            vision,
            "--output",
            results / "real-task-acceptance-cpu.json",
        ],
    )
    payload = {
        "status": "running",
        "jobs": [],
        "order": [name for name, _ in jobs],
        "methodology": "serial jobs; full references; separate-process alternating-order timing replicates; capped large-model execution explicitly labeled stress",
    }
    status = results / "campaign-status.json"
    if args.start_at:
        names = [name for name, _ in jobs]
        if args.start_at not in names:
            raise ValueError(f"unknown campaign job: {args.start_at}")
        index = names.index(args.start_at)
        if status.exists():
            previous = json.loads(status.read_text())
            payload["jobs"] = [
                entry for entry in previous["jobs"] if entry["name"] in names[:index]
            ]
            payload["previous_attempts"] = previous.get("previous_attempts", []) + [previous]
        jobs = jobs[index:]
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "8",
        "OMP_NUM_THREADS": "8",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    for name, command in jobs:
        print(f"CAMPAIGN START {name}", flush=True)
        entry = {
            "name": name,
            "command": command,
            "status": "running",
            "log": str(log_dir / f"{name}.log"),
        }
        log_path = Path(entry["log"])
        attempt = 1
        while log_path.exists():
            attempt += 1
            log_path = log_dir / f"{name}-attempt{attempt}.log"
        entry["log"] = str(log_path)
        payload["jobs"].append(entry)
        status.write_text(json.dumps(payload, indent=2) + "\n")
        started = time.perf_counter()
        with (
            Path(entry["log"]).open("w") as log,
            subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ) as child,
        ):
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(f"[{name}] {line}", end="", flush=True)
            code = child.wait()
        entry.update(
            status="complete" if code == 0 else "failed",
            exit_code=code,
            seconds=time.perf_counter() - started,
        )
        status.write_text(json.dumps(payload, indent=2) + "\n")
        if code != 0:
            payload["status"] = "failed"
            status.write_text(json.dumps(payload, indent=2) + "\n")
            raise SystemExit(code)
    payload["status"] = "complete"
    status.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait-for",
        type=Path,
        help="Wait for the current TensorSmith GPU-training report to become complete before starting any new job",
    )
    parser.add_argument(
        "--start-at",
        help="Explicit recovery entry point after diagnosing a failed job; preserves previous status/log attempts",
    )
    parser.add_argument(
        "--refresh-tensorsmith",
        action="store_true",
        help="Add full TensorSmith retraining before acceptance/timing after a runtime fix; archive previous reports/checkpoints first",
    )
    run(parser.parse_args())
