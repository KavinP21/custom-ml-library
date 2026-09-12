"""End-to-end CIFAR-10 / WikiText-2 training, held-out evaluation and timing.

This is NOT a certified MLPerf submission or a SOTA comparison. No synthetic
substitutes, training-set quality metrics, or silently reduced datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from task_data import DEFAULT_CACHE, image_batch, language_batches, load_cifar, load_wikitext
from task_models import build_model, torch_reference

import tensorsmith as ts
from tensorsmith import nn, optim


def source_digest():
    value = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted(
        list((root / "src").rglob("*.py")) + list((root / "benchmarks").glob("*.py"))
    ):
        value.update(str(path.relative_to(root)).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def batches(task, split, args, epoch=None):
    if task == "wikitext2":
        yield from language_batches(
            split, args.batch_size if epoch is not None else args.eval_batch_size, args.seq_len
        )
    else:
        x, y = split
        rng = np.random.default_rng(args.seed + epoch) if epoch is not None else None
        indices = rng.permutation(len(x)) if rng is not None else np.arange(len(x))
        size = args.batch_size if epoch is not None else args.eval_batch_size
        for start in range(0, len(x), size):
            ids = indices[start : start + size]
            yield image_batch(x[ids], rng), y[ids]


class Engine:
    def __init__(self, model, args):
        self.args = args
        self.is_ts = args.engine == "tensorsmith"
        self.device = args.device
        if self.is_ts:
            self.model = model.to(args.device)
            cls = optim.SGD if args.task == "cifar10" else optim.AdamW
            keywords = (
                {"momentum": 0.9}
                if args.task == "cifar10"
                else {"betas": (0.9, 0.999), "eps": 1e-8}
            )
        else:
            import torch

            self.torch = torch
            torch.set_num_threads(args.threads)
            torch.manual_seed(args.seed)
            self.device = "mps" if args.device == "metal" else args.device
            if self.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("PyTorch MPS is unavailable; CPU fallback is prohibited")
            self.model = torch_reference(args.task, model, self.device)
            cls = torch.optim.SGD if args.task == "cifar10" else torch.optim.AdamW
            keywords = (
                {"momentum": 0.9}
                if args.task == "cifar10"
                else {"betas": (0.9, 0.999), "eps": 1e-8, "foreach": False}
            )
        self.parameters = list(self.model.parameters())
        self.optimizer = cls(
            self.parameters, lr=args.lr, weight_decay=args.weight_decay, **keywords
        )
        self.completed_updates = 0

    def transfer(self, x, y):
        if self.is_ts:
            return ts.tensor(x, device=self.device), ts.tensor(y, device=self.device)
        return self.torch.tensor(x, device=self.device), self.torch.tensor(y, device=self.device)

    def synchronize(self, values=()):
        if self.is_ts:
            ts.evaluate(values)
            ts.synchronize(self.device)
        elif self.device == "mps":
            self.torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def loss(self, logits, labels):
        if self.is_ts:
            return nn.functional.cross_entropy(logits, labels, axis=-1)
        return self.torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )

    def train_step(self, x, y, lr):
        self.model.train()
        self.optimizer.zero_grad()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        inputs, labels = self.transfer(x, y)
        logits = self.model(inputs)
        loss = self.loss(logits, labels)
        loss.backward()
        if self.args.task == "wikitext2":
            if self.is_ts:
                nn.utils.clip_grad_norm_(self.parameters, 1.0)
            else:
                self.torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, foreach=False)
        self.optimizer.step()
        self.completed_updates += 1
        values = [loss]
        if self.is_ts:
            # Complete the entire update, not merely a queued forward loss.
            values += [p._data for p in self.parameters] + [p._grad for p in self.parameters]
            values += list(self.optimizer.state.values())
            values += [
                value
                for _, module in self.model.named_modules()
                for value in module._buffers.values()
            ]
        self.synchronize(values)
        return float(loss.item())

    def evaluate(self, split):
        self.model.eval()
        context = ts.no_grad() if self.is_ts else self.torch.no_grad()
        started = time.perf_counter()
        total_loss, total, correct = 0.0, 0, 0
        with context:
            for x, y in batches(self.args.task, split, self.args):
                inputs, labels = self.transfer(x, y)
                logits = self.model(inputs)
                loss = self.loss(logits, labels)
                self.synchronize([loss, logits])
                n = y.size
                total_loss += float(loss.item()) * n
                total += n
                if self.args.task == "cifar10":
                    if self.is_ts:
                        from tensorsmith.device import xp_for

                        correct += int(
                            xp_for(self.device).sum(
                                xp_for(self.device).argmax(logits._data, axis=-1) == labels._data
                            )
                        )
                    else:
                        correct += int((logits.argmax(-1) == labels).sum().item())
        nll = total_loss / total
        result = {"nll": nll, "evaluated_targets": total, "seconds": time.perf_counter() - started}
        if self.args.task == "wikitext2":
            result["perplexity"] = math.exp(nll)
        else:
            result["accuracy"] = correct / total
        return result

    def state(self):
        if self.is_ts:
            return self.model.state_dict()
        result = {name: value.detach().cpu().numpy() for name, value in self.model.weights.items()}
        result.update(
            {
                name: value.detach().cpu().numpy()
                for name, value in self.model.buffers_by_name.items()
            }
        )
        return result

    def load_model(self, state):
        if self.is_ts:
            self.model.load_state_dict(state)
        else:
            with self.torch.no_grad():
                for name, value in {**self.model.weights, **self.model.buffers_by_name}.items():
                    value.copy_(self.torch.tensor(state[name], device=self.device))


def learning_rate(args, step, total_steps):
    warmup = min(args.warmup_steps, total_steps - 1)
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return args.min_lr + (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress)) / 2


def timing(samples, units):
    return {
        "completed_steps": len(samples),
        "median_ms": statistics.median(samples) * 1000,
        "mean_ms": statistics.mean(samples) * 1000,
        "p95_ms": float(np.percentile(samples, 95)) * 1000,
        "samples_ms": [s * 1000 for s in samples],
        "units_per_second": sum(units) / sum(samples),
    }


def run(args):
    if args.task == "cifar10":
        splits, provenance = load_cifar(args.cache, args.seed)
        vocabulary = None
        sizes = {key: len(value[1]) for key, value in splits.items()}
    else:
        splits, vocabulary, provenance = load_wikitext(args.cache)
        sizes = {key: len(value) for key, value in splits.items()}
    model = build_model(args.task, args, None if vocabulary is None else len(vocabulary))
    engine = Engine(model, args)
    engine.synchronize(list(engine.parameters))
    parameter_count = sum(int(np.prod(p.shape)) for p in model.parameters())
    steps_per_epoch = sum(1 for _ in batches(args.task, splits["train"], args, epoch=0))
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total_steps = steps_per_epoch * args.epochs
    versions = {}
    for name in ("tensorsmith", "numpy", "mlx", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    hardware = {}
    if args.device == "metal":
        from tensorsmith.device import xp_for

        hardware = xp_for("metal").device_info()
    payload = {
        "status": "running",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_digest(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": versions,
            "hardware": hardware,
        },
        "dataset": {
            "provenance": provenance,
            "split_sizes": sizes,
            "vocabulary_size": None if vocabulary is None else len(vocabulary),
        },
        "model": {
            "parameter_count": parameter_count,
            "architecture": "CIFAR ResNet20 (projection shortcuts)"
            if args.task == "cifar10" and args.blocks_per_stage == 3
            else "CIFAR residual CNN"
            if args.task == "cifar10"
            else "decoder-only GQA/RoPE/RMSNorm/SwiGLU transformer; tied embedding",
        },
        "methodology": {
            "quality": "full validation and full test; best checkpoint selected by validation only",
            "timing": "wall-clock completed training updates; includes CPU augmentation, input transfer, forward, loss, backward, gradient clipping (LM), optimizer and synchronization; first warmup steps excluded from timing only, not training",
            "language_protocol": "train-only whitespace vocabulary; OOV-><unk>; <eos> after every line; fixed streams; attention context resets at each segment; token-weighted NLL, natural-log perplexity; not a leaderboard-identical sliding-window protocol",
            "certification": "research task benchmark, NOT MLPerf-certified; single seed unless separately repeated",
        },
        "initial_validation": engine.evaluate(splits["valid"]),
        "epochs": [],
    }
    payload["initial_state_sha256"] = hashlib.sha256(
        b"".join(name.encode() + value.tobytes() for name, value in model.state_dict().items())
    ).hexdigest()
    if vocabulary is not None:
        counts = np.bincount(splits["train"], minlength=len(vocabulary)).astype(np.float64) + 1
        probability = counts / counts.sum()
        # Baseline fitted solely on training counts, scored on validation.
        payload["train_fitted_unigram_validation_perplexity"] = float(
            np.exp(-np.log(probability[splits["valid"]]).mean())
        )
    checkpoint = (
        args.checkpoint
        or Path("benchmarks/checkpoints") / f"{args.task}-{args.engine}-{args.device}.npz"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save_report():
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    save_report()
    print(
        f"{args.task} {args.engine} {args.device}: {parameter_count:,} parameters; splits={sizes}; initial_validation={payload['initial_validation']}",
        flush=True,
    )
    best_nll = float("inf")
    all_samples, all_units = [], []
    wall_start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        total_loss, total_targets = 0.0, 0
        samples, units = [], []
        iterator = iter(batches(args.task, splits["train"], args, epoch))
        for step in range(steps_per_epoch):
            started = time.perf_counter()
            x, y = next(iterator)
            loss = engine.train_step(
                x, y, learning_rate(args, engine.completed_updates, total_steps)
            )
            elapsed = time.perf_counter() - started
            if not math.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch + 1}, update {step + 1}"
                )
            total_loss += loss * y.size
            total_targets += y.size
            if step >= args.timing_warmup:
                samples.append(elapsed)
                units.append(y.size)
            if step == 0 or (step + 1) % args.log_interval == 0 or step + 1 == steps_per_epoch:
                print(
                    f"epoch={epoch + 1}/{args.epochs} update={step + 1}/{steps_per_epoch} loss={total_loss / total_targets:.4f} last_ms={elapsed * 1000:.1f}",
                    flush=True,
                )
        train_seconds = time.perf_counter() - epoch_start
        validation = engine.evaluate(splits["valid"])
        entry = {
            "epoch": epoch + 1,
            "training_nll": total_loss / total_targets,
            "training_targets": total_targets,
            "training_seconds": train_seconds,
            "validation": validation,
            "timing": timing(samples, units) if samples else None,
        }
        payload["epochs"].append(entry)
        all_samples.extend(samples)
        all_units.extend(units)
        if validation["nll"] < best_nll:
            best_nll = validation["nll"]
            payload["best_epoch"] = epoch + 1
            # Common safe model checkpoint format, usable across both engines.
            ts.save(
                {
                    "model": engine.state(),
                    "epoch": epoch + 1,
                    "config": payload["arguments"],
                    "vocabulary": vocabulary,
                    "optimizer": engine.optimizer.state_dict() if engine.is_ts else None,
                },
                checkpoint,
            )
        save_report()
        print(
            f"epoch={epoch + 1} validation={validation} training_seconds={train_seconds:.1f}",
            flush=True,
        )
    engine.load_model(ts.load(checkpoint)["model"])
    payload["test"] = engine.evaluate(splits["test"])
    payload["training_timing"] = timing(all_samples, all_units) if all_samples else None
    payload["total_training_targets"] = sum(e["training_targets"] for e in payload["epochs"])
    payload["training_and_evaluation_seconds"] = time.perf_counter() - wall_start
    payload["checkpoint"] = str(checkpoint)
    payload["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    payload["memory_note"] = (
        "process high-water resident memory; includes data, CPU/runtime and unified-memory allocations; not comparable to allocator-only GPU peak"
    )
    if engine.is_ts and args.device == "metal":
        from tensorsmith.device import xp_for

        mx = xp_for("metal")
        payload["mlx_allocator_peak_bytes"] = mx.get_peak_memory()
    payload["status"] = "complete"
    save_report()
    print(
        f"COMPLETE best_epoch={payload['best_epoch']} test={payload['test']} report={args.output}",
        flush=True,
    )
    return payload


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["cifar10", "wikitext2"], required=True)
    p.add_argument("--engine", choices=["tensorsmith", "torch"], default="tensorsmith")
    p.add_argument("--device", choices=["cpu", "metal", "cuda"], default="cpu")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--min-lr", type=float, default=0.0001)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--timing-warmup", type=int, default=5)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Explicit diagnostic cap per epoch; 0 means FULL training split",
    )
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--blocks-per-stage", type=int, default=3)
    p.add_argument("--width", type=int, default=16)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if (
        min(
            args.epochs,
            args.batch_size,
            args.eval_batch_size,
            args.seq_len,
            args.log_interval,
            args.threads,
        )
        <= 0
        or args.max_steps < 0
        or args.timing_warmup < 0
        or args.warmup_steps < 0
        or not 0 <= args.min_lr <= args.lr
    ):
        raise ValueError("invalid task benchmark arguments")
    if args.engine == "torch" and args.activation_checkpointing:
        raise ValueError(
            "reference checkpointing is not implemented; use matched non-checkpointed runs"
        )
    run(args)
