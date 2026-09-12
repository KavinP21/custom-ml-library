"""Isolated, repeated complete-model timing on real dataset minibatches.

Run each engine in a SEPARATE process, without another GPU job. Training is
optimizer-inclusive; inference materializes every requested output. Dataset
loading and compilation warmup are outside the measured region.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

from bench_tasks import Engine, source_digest
from common import measure, report
from task_data import DEFAULT_CACHE, image_batch, language_batches, load_cifar, load_wikitext
from task_models import build_model

import tensorsmith as ts


def run(args):
    saved = ts.load(args.checkpoint)
    config = {
        **saved["config"],
        "engine": args.engine,
        "device": args.device,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "threads": 8,
    }
    config["activation_checkpointing"] = False
    options = SimpleNamespace(**config)
    task = config["task"]
    if task == "cifar10":
        splits, _ = load_cifar(args.cache, config["seed"])
        x = image_batch(splits["valid"][0][: args.batch_size])
        y = splits["valid"][1][: args.batch_size]
    else:
        splits, _, _ = load_wikitext(args.cache)
        x, y = next(language_batches(splits["valid"], args.batch_size, args.seq_len))
    model = build_model(
        task, options, None if saved["vocabulary"] is None else len(saved["vocabulary"])
    )
    model.load_state_dict(saved["model"])
    engine = Engine(model, options)
    if args.torch_fused_optimizer:
        if engine.is_ts or task != "wikitext2":
            raise ValueError("the optional fused baseline is PyTorch AdamW for WikiText-2 only")
        # In addition to the matched eager/default reference, expose the
        # available native fused AdamW variant rather than silently handicapping
        # an optimized baseline. No fallback on unsupported hardware is allowed.
        engine.optimizer = engine.torch.optim.AdamW(
            engine.parameters,
            lr=options.lr,
            weight_decay=options.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            foreach=False,
            fused=True,
        )
    inputs, labels = engine.transfer(x, y)
    engine.synchronize([inputs, labels])
    results = []

    def memory():
        if engine.is_ts and args.device == "metal":
            from tensorsmith.device import xp_for

            mx = xp_for("metal")
            return {
                "mlx_active_bytes": mx.get_active_memory(),
                "mlx_cache_bytes": mx.get_cache_memory(),
                "mlx_peak_active_bytes": mx.get_peak_memory(),
            }
        if not engine.is_ts and engine.device == "mps":
            return {
                "torch_mps_current_allocated_bytes": engine.torch.mps.current_allocated_memory(),
                "torch_mps_driver_allocated_bytes": engine.torch.mps.driver_allocated_memory(),
                "peak_unavailable": True,
            }
        return {}

    def reset_peak():
        if engine.is_ts and args.device == "metal":
            from tensorsmith.device import xp_for

            xp_for("metal").reset_peak_memory()

    reset_peak()
    results.append(
        measure(
            "real_data_complete_training_update",
            args.device,
            lambda: engine.train_step(x, y, 1e-5),
            args.iterations,
            args.warmup,
            units=y.size,
            completion=lambda _d, v: engine.synchronize(v),
        )
    )
    results[-1]["memory_after_phase"] = memory()
    # Restore the same trained weights before measuring inference.
    engine.optimizer.zero_grad()
    engine.optimizer.state.clear()
    engine.load_model(saved["model"])
    gc.collect()
    engine.model.eval()
    context = ts.no_grad() if engine.is_ts else engine.torch.no_grad()
    with context:
        reset_peak()
        results.append(
            measure(
                "real_data_full_logits_inference",
                args.device,
                lambda: engine.model(inputs),
                args.iterations,
                args.warmup,
                units=y.size,
                completion=lambda _d, v: engine.synchronize(v),
            )
        )
        results[-1]["memory_after_phase"] = memory()
        if engine.is_ts and task == "wikitext2":
            # Full vocabulary logits of only the newest token, same output
            # semantics for cached/uncached paths, with exact parity guard.
            prompt = ts.tensor(splits["valid"][:512].reshape(1, -1), device=args.device)
            next_token = ts.tensor(splits["valid"][512:513].reshape(1, 1), device=args.device)
            cache = engine.model.new_cache(1, max_seq_len=768)
            engine.synchronize(engine.model(prompt, caches=cache, logits_to_keep=1))
            expected = engine.model(ts.cat([prompt, next_token], dim=1), logits_to_keep=1).numpy()
            actual = engine.model(next_token, caches=cache).numpy()
            import numpy as np

            np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=2e-3)

            def decode():
                for c in cache:
                    c.truncate(512)
                logits = engine.model(next_token, caches=cache)
                return [logits, [c.storage for c in cache]]

            results.append(
                measure(
                    "real_text_512_prefix_cached_decode",
                    args.device,
                    decode,
                    args.iterations,
                    args.warmup,
                    units=1,
                    completion=lambda _d, v: engine.synchronize(v),
                )
            )
            results.append(
                measure(
                    "real_text_512_prefix_uncached_decode",
                    args.device,
                    lambda: engine.model(ts.cat([prompt, next_token], dim=1), logits_to_keep=1),
                    args.iterations,
                    args.warmup,
                    units=1,
                    completion=lambda _d, v: engine.synchronize(v),
                )
            )
    args.json = str(args.output)
    payload = report(results, args)
    # common.report writes a payload but currently returns nothing.
    if payload is None:
        payload = json.loads(args.output.read_text())
    payload["source_sha256"] = source_digest()
    payload["engine"] = args.engine
    payload["dataset_input_shape"] = list(x.shape)
    payload["quality_checkpoint"] = str(args.checkpoint)
    payload["memory_note"] = (
        "MLX peak-active includes phase warmup and transient allocations; Torch MPS current/driver are end-of-phase values, not peak. Do not compare these as equivalent GPU memory metrics. Optimizer/gradients released before inference."
    )
    payload["methodology"] += (
        "; task models loaded from trained checkpoints; training includes host-to-device input transfer but not CPU augmentation; inference inputs resident on device; engines run separately; cached timings reuse a fixed real prefix, not a growing-cache serving benchmark"
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--engine", choices=["tensorsmith", "torch"], default="tensorsmith")
    parser.add_argument("--device", choices=["cpu", "metal", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--torch-fused-optimizer",
        action="store_true",
        help="Additional PyTorch language baseline using native fused AdamW; fail if unsupported",
    )
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if min(options.batch_size, options.seq_len, options.iterations) <= 0 or options.warmup < 0:
        raise ValueError("invalid timing configuration")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    run(options)
