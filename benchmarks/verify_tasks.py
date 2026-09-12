"""Independent real-data acceptance checks for complete models and stateful paths.

Run AFTER task training. Every parameter gradient is checked, not a sampled
handful. Failures are recorded and cause a nonzero exit status.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from bench_tasks import source_digest
from task_data import DEFAULT_CACHE, image_batch, language_batches, load_cifar, load_wikitext
from task_models import build_model, torch_reference

import tensorsmith as ts
from tensorsmith import nn, optim


def discrepancy(actual, expected, *, atol, rtol, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"{label}: non-finite values")
    error = actual.astype(np.float64) - expected.astype(np.float64)
    stats = {
        "max_absolute_error": float(np.max(np.abs(error))),
        "relative_l2_error": float(
            np.linalg.norm(error.reshape(-1)) / max(np.linalg.norm(expected.reshape(-1)), 1e-12)
        ),
        "elements": int(actual.size),
    }
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)
    return stats


def clone_model(saved, task, device, *, dtype="float32", checkpointing=False):
    args = SimpleNamespace(**saved["config"])
    model = build_model(
        task, args, None if saved["vocabulary"] is None else len(saved["vocabulary"])
    )
    model.load_state_dict(saved["model"])
    model.to(device, dtype=dtype)
    if task == "wikitext2" and checkpointing:
        model.config = replace(model.config, activation_checkpointing=True)
    return model


def complete(model, optimizer=None, values=()):
    ts.evaluate(
        values,
        [p._data for p in model.parameters()],
        [p._grad for p in model.parameters()],
        None if optimizer is None else list(optimizer.state.values()),
        [v for _, module in model.named_modules() for v in module._buffers.values()],
    )
    ts.synchronize(next(model.parameters()).device)


def oracle(saved, task, device, real_batches):
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(8)
    target_device = "mps" if device == "metal" else device
    if target_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no fallback permitted")
    actual = clone_model(saved, task, device)
    if task == "wikitext2":
        longest = max(x.shape[1] for x, _ in real_batches)
        actual.config = replace(actual.config, max_seq_len=max(actual.config.max_seq_len, longest))
    expected = torch_reference(task, actual, target_device)
    if task == "cifar10":
        aopt = optim.SGD(actual.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        eopt = torch.optim.SGD(expected.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    else:
        aopt = optim.AdamW(actual.parameters(), lr=0.0001, weight_decay=0.01)
        eopt = torch.optim.AdamW(expected.parameters(), lr=0.0001, weight_decay=0.01, foreach=False)
    results = []
    for step, (x, y) in enumerate(real_batches):
        aopt.zero_grad()
        eopt.zero_grad()
        logits = actual(ts.tensor(x, device=device))
        expected_logits = expected(torch.tensor(x, device=target_device))
        aloss = nn.functional.cross_entropy(logits, ts.tensor(y, device=device), axis=-1)
        eloss = F.cross_entropy(
            expected_logits.reshape(-1, expected_logits.shape[-1]),
            torch.tensor(y, device=target_device).reshape(-1),
        )
        aloss.backward()
        eloss.backward()
        entry = {
            "step": step + 1,
            "input_shape": list(x.shape),
            "logits": discrepancy(
                logits.numpy(),
                expected_logits.detach().cpu().numpy(),
                atol=2e-4,
                rtol=2e-3,
                label=f"{task} logits",
            ),
            "loss_error": abs(aloss.item() - eloss.item()),
            "gradients": {},
        }
        if entry["loss_error"] > 3e-5:
            raise AssertionError("independent task loss mismatch")
        for name, p in actual.named_parameters():
            if p.grad is None or expected.weights[name].grad is None:
                raise AssertionError(f"missing parameter gradient: {name}")
            entry["gradients"][name] = discrepancy(
                p.grad.numpy(),
                expected.weights[name].grad.detach().cpu().numpy(),
                atol=3e-5,
                rtol=4e-3,
                label=f"{task} gradient {name}",
            )
            expected_gradient = expected.weights[name].grad.detach().cpu().numpy()
            if (
                np.linalg.norm(expected_gradient.reshape(-1)) > 1e-6
                and entry["gradients"][name]["relative_l2_error"] > 0.005
            ):
                raise AssertionError(
                    f"{task} gradient norm disagreement: {name}: {entry['gradients'][name]}"
                )
        # Isolate optimizer arithmetic from Adam's near-zero-gradient sign
        # sensitivity. Differentiation was independently checked above.
        for name, p in actual.named_parameters():
            expected.weights[name].grad = torch.tensor(p.grad.numpy(), device=target_device)
        entry["optimizer_oracle"] = (
            "identical TensorSmith gradients supplied to both optimizers AFTER independent torch-gradient checks"
        )
        aopt.step()
        eopt.step()
        complete(actual, aopt)
        entry["updated_weights"] = {}
        entry["optimizer_state"] = {}
        for name, p in actual.named_parameters():
            entry["updated_weights"][name] = discrepancy(
                p.numpy(),
                expected.weights[name].detach().cpu().numpy(),
                atol=3e-6,
                rtol=3e-5,
                label=f"{task} optimizer {name}",
            )
            entry["optimizer_state"][name] = {}
            for key, value in aopt.state[id(p)].items():
                reference_state = eopt.state[expected.weights[name]][key]
                if hasattr(value, "shape"):
                    entry["optimizer_state"][name][key] = discrepancy(
                        ts.tensor(value, device=device).numpy(),
                        reference_state.detach().cpu().numpy(),
                        atol=3e-7,
                        rtol=3e-4,
                        label=f"{task} optimizer state {name}/{key}",
                    )
                elif value != int(reference_state.item()):
                    raise AssertionError(f"optimizer step counter mismatch: {name}")
        for name, value in expected.buffers_by_name.items():
            discrepancy(
                actual.state_dict()[name],
                value.cpu().numpy(),
                atol=1e-5,
                rtol=2e-4,
                label=f"{task} running state {name}",
            )
        results.append(entry)
    return {
        "batches": results,
        "checked_parameters_per_batch": len(list(actual.parameters())),
        "checked_gradient_elements_per_batch": sum(p.numel() for p in actual.parameters()),
    }


def checkpoint_accumulation(saved, device, batch):
    x, y = batch
    ordinary = clone_model(saved, "wikitext2", device)
    checkpointed = clone_model(saved, "wikitext2", device, checkpointing=True)
    accumulated = clone_model(saved, "wikitext2", device)
    inp, labels = ts.tensor(x, device=device), ts.tensor(y, device=device)
    loss = ordinary.loss(inp, labels)
    loss.backward()
    replay_loss = checkpointed.loss(inp, labels)
    replay_loss.backward()
    for index in range(len(x)):
        micro_loss = accumulated.loss(inp[index : index + 1], labels[index : index + 1]) / len(x)
        micro_loss.backward()
    result = {
        "loss_error": abs(loss.item() - replay_loss.item()),
        "checkpointed": {},
        "accumulated": {},
    }
    for (name, p), replay, accum in zip(
        ordinary.named_parameters(), checkpointed.parameters(), accumulated.parameters()
    ):
        result["checkpointed"][name] = discrepancy(
            replay.grad.numpy(), p.grad.numpy(), atol=2e-5, rtol=3e-3, label=f"checkpoint {name}"
        )
        result["accumulated"][name] = discrepancy(
            accum.grad.numpy(), p.grad.numpy(), atol=2e-5, rtol=3e-3, label=f"accumulation {name}"
        )
    return result


def attention_paths(saved, device, batch):
    x, y = batch
    models = [clone_model(saved, "wikitext2", device) for _ in range(3)]
    names = ["auto", "dense", "streaming"]
    results = []
    for name, model in zip(names, models):
        for block in model.blocks:
            block.attention.backend = name
        loss = model.loss(ts.tensor(x, device=device), ts.tensor(y, device=device))
        loss.backward()
        complete(model, values=[loss])
        results.append(loss.item())
    checked = {}
    for name, alternative in zip(names[1:], models[1:]):
        checked[name] = {}
        for (parameter_name, p), reference in zip(
            alternative.named_parameters(), models[0].parameters()
        ):
            checked[name][parameter_name] = discrepancy(
                p.grad.numpy(),
                reference.grad.numpy(),
                atol=2e-5,
                rtol=3e-3,
                label=f"attention path {name} {parameter_name}",
            )
    return {"input_shape": list(x.shape), "losses": dict(zip(names, results)), "gradients": checked}


def stochastic_checkpoint(saved, device, batch):
    modified = {**saved, "config": {**saved["config"], "dropout": 0.1}}
    ordinary = clone_model(modified, "wikitext2", device)
    replay = clone_model(modified, "wikitext2", device, checkpointing=True)
    x, y = batch
    ts.seed(981)
    aloss = ordinary.loss(ts.tensor(x, device=device), ts.tensor(y, device=device))
    aloss.backward()
    complete(ordinary, values=[aloss])
    from tensorsmith.device import random_uniform

    # This is an explicit oracle export, just like logits/gradient .numpy()
    # above, not an operator's prohibited host compatibility fallback.
    next_random = ts.tensor(random_uniform((32,), device), device=device).numpy()
    ts.seed(981)
    bloss = replay.loss(ts.tensor(x, device=device), ts.tensor(y, device=device))
    bloss.backward()
    complete(replay, values=[bloss])
    after_random = ts.tensor(random_uniform((32,), device), device=device).numpy()
    np.testing.assert_array_equal(
        after_random, next_random, err_msg="checkpoint changed next RNG draw"
    )
    if abs(aloss.item() - bloss.item()) > 1e-5:
        raise AssertionError("stochastic checkpoint loss mismatch")
    result = {}
    for (name, p), q in zip(ordinary.named_parameters(), replay.parameters()):
        result[name] = discrepancy(
            q.grad.numpy(),
            p.grad.numpy(),
            atol=2e-5,
            rtol=3e-3,
            label=f"stochastic checkpoint {name}",
        )
    return {
        "dropout": 0.1,
        "loss_error": abs(aloss.item() - bloss.item()),
        "rng_continuation_exact": True,
        "gradients": result,
    }


def resume(saved, task, device, real_batches):
    model = clone_model(saved, task, device)
    cls = optim.AdamW if task == "wikitext2" else optim.SGD
    kwargs = {"momentum": 0.9} if task == "cifar10" else {}
    optimizer = cls(model.parameters(), lr=0.0001, **kwargs)
    scheduler = optim.WarmupCosineLR(optimizer, warmup_steps=1, total_steps=8, min_lr=1e-5)

    def update(m, opt, sched, batch):
        x, y = batch
        opt.zero_grad()
        loss = nn.functional.cross_entropy(
            m(ts.tensor(x, device=device)), ts.tensor(y, device=device), axis=-1
        )
        loss.backward()
        opt.step()
        sched.step()
        complete(m, opt, [loss])
        return loss.item()

    update(model, optimizer, scheduler, real_batches[0])
    with tempfile.TemporaryDirectory(prefix="tensorsmith-task-resume-") as directory:
        path = Path(directory) / "checkpoint.npz"
        ts.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            path,
        )
        restored = ts.load(path)
    resumed = clone_model(saved, task, device)
    resumed.load_state_dict(restored["model"])
    resumed_opt = cls(resumed.parameters(), lr=0.0001, **kwargs)
    resumed_opt.load_state_dict(restored["optimizer"])
    resumed_sched = optim.WarmupCosineLR(resumed_opt, 1, 8, min_lr=1e-5)
    resumed_sched.load_state_dict(restored["scheduler"])
    errors = []
    for batch in real_batches[1:]:
        a, b = (
            update(model, optimizer, scheduler, batch),
            update(resumed, resumed_opt, resumed_sched, batch),
        )
        errors.append(abs(a - b))
    checked = {}
    for name, value in model.state_dict().items():
        checked[name] = discrepancy(
            resumed.state_dict()[name], value, atol=2e-6, rtol=2e-5, label=f"resume {name}"
        )
    left, right = optimizer.state_dict(), resumed_opt.state_dict()
    state_arrays = 0
    for index, state in left["state"].items():
        for name, value in state.items():
            incoming = right["state"][index][name]
            if hasattr(value, "shape"):
                discrepancy(
                    incoming, value, atol=2e-7, rtol=2e-5, label=f"resume optimizer {index}/{name}"
                )
                state_arrays += 1
            elif value != incoming:
                raise AssertionError("optimizer scalar state mismatch")
    if scheduler.state_dict() != resumed_sched.state_dict():
        raise AssertionError("scheduler state mismatch")
    return {
        "continuation_loss_errors": errors,
        "model_and_buffers": checked,
        "optimizer_state_arrays": state_arrays,
        "continued_updates": len(real_batches) - 1,
    }


def cached_language(saved, device, tokens):
    model = clone_model(saved, "wikitext2", device).eval()
    raw = tokens[: 2 * 640].reshape(2, 640)
    ids = ts.tensor(raw, device=device)
    targets = ts.tensor(tokens[1 : 2 * 640 + 1].reshape(2, 640), device=device)
    report = {}
    with ts.no_grad():
        full = model(ids)
        baseline_nll = nn.functional.cross_entropy(full, targets, axis=-1).item()
        for quantized in (False, True):
            cache = model.new_cache(2, max_seq_len=768, quantized=quantized)
            parts = []
            start = 0
            # Unequal chunks, not simply prefill followed by one toy token.
            for length in (257, 1, 31, 127, 224):
                logits = model(ids[:, start : start + length], caches=cache)
                ts.evaluate(logits, [c.storage for c in cache])
                parts.append(logits)
                start += length
            merged = ts.cat(parts, dim=1)
            nll = nn.functional.cross_entropy(merged, targets, axis=-1).item()
            name = "int8" if quantized else "float32"
            if quantized:
                if abs(nll - baseline_nll) > 0.03:
                    raise AssertionError(f"quantized cache NLL degradation {nll - baseline_nll}")
                error = {"max_absolute_error": float(np.max(np.abs(merged.numpy() - full.numpy())))}
            else:
                error = discrepancy(
                    merged.numpy(),
                    full.numpy(),
                    atol=3e-4,
                    rtol=2e-3,
                    label="640-token chunked cache",
                )
            # Reorder/duplicate beams then compare against independent prefix.
            for c in cache:
                c.reorder(np.array([1, 1], np.int32))
            next_raw = np.repeat(tokens[1280:1281].reshape(1, 1), 2, axis=0)
            next_ids = ts.tensor(next_raw, device=device)
            cached_next = model(next_ids, caches=cache)
            direct_next = model(
                ts.cat([ids[ts.tensor([1, 1], device=device)], next_ids], dim=1), logits_to_keep=1
            )
            if not quantized:
                discrepancy(
                    cached_next.numpy(),
                    direct_next.numpy(),
                    atol=3e-4,
                    rtol=2e-3,
                    label="reordered beams",
                )
            else:
                next_label = ts.tensor(
                    np.repeat(tokens[1281:1282].reshape(1, 1), 2, axis=0), device=device
                )
                cached_nll = nn.functional.cross_entropy(cached_next, next_label, axis=-1).item()
                direct_nll = nn.functional.cross_entropy(direct_next, next_label, axis=-1).item()
                if abs(cached_nll - direct_nll) > 0.03:
                    raise AssertionError("int8 reordered-beam NLL drift")
            report[name] = {
                "chunk_lengths": [257, 1, 31, 127, 224],
                "nll": nll,
                "nll_difference": nll - baseline_nll,
                "logits_error": error,
                "persistent_cache_bytes": sum(c.memory_bytes for c in cache),
                "beam_reorder_exercised": True,
            }
        prompt = ids[:1, :128]
        cached = model.generate(prompt, max_new_tokens=64, temperature=0, use_cache=True)
        uncached = model.generate(prompt, max_new_tokens=64, temperature=0, use_cache=False)
        np.testing.assert_array_equal(
            cached.numpy(), uncached.numpy(), err_msg="64-token autoregressive cache equivalence"
        )
        report["greedy_generated_tokens_match"] = 64
        vocabulary = saved["vocabulary"]
        report["generated_words"] = " ".join(vocabulary[i] for i in cached.numpy()[0, -64:])
    return report


def low_precision(saved, device, real_batches):
    full = clone_model(saved, "wikitext2", device).eval()
    low = clone_model(saved, "wikitext2", device, dtype="float16").eval()
    baseline, quantized, total = 0.0, 0.0, 0
    with ts.no_grad():
        for x, y in real_batches:
            ids, labels = ts.tensor(x, device=device), ts.tensor(y, device=device)
            baseline += full.loss(ids, labels).item() * y.size
            quantized += low.loss(ids, labels).item() * y.size
            total += y.size
    if abs(quantized / total - baseline / total) > 0.015:
        raise AssertionError("FP16 held-out NLL drift exceeds acceptance threshold")
    low.train()
    optimizer = optim.AdamW(low.parameters(), lr=1e-5)
    scaler = ts.amp.GradScaler(init_scale=128)
    losses, skipped = [], 0
    for x, y in real_batches * 2:
        optimizer.zero_grad()
        loss = low.loss(ts.tensor(x, device=device), ts.tensor(y, device=device))
        scaler.scale(loss).backward()
        finite = scaler.unscale_(optimizer)
        if finite:
            nn.utils.clip_grad_norm_(low.parameters(), 1.0)
        applied = scaler.step(optimizer)
        scaler.update()
        skipped += not applied
        complete(low, optimizer, [loss])
        value = loss.item()
        if not math.isfinite(value):
            raise AssertionError("nonfinite FP16 task loss")
        losses.append(value)
    if skipped:
        raise AssertionError(f"unexpected overflow on these FP16 acceptance batches: {skipped}")
    for p in low.parameters():
        if not np.isfinite(p.numpy()).all():
            raise AssertionError("nonfinite FP16 parameter")
    return {
        "held_out_targets": total,
        "fp32_nll": baseline / total,
        "fp16_nll": quantized / total,
        "fp16_training_losses": losses,
        "skipped_updates": skipped,
        "loss_scale": scaler.state_dict()["scale"],
    }


def mixed_operator_graph(device, raw_images):
    """Unused-by-ResNet ops compose over genuine image signals, with a torch oracle."""
    import torch
    from torch.nn import functional as F

    target = "mps" if device == "metal" else device
    rng = np.random.default_rng(52)
    raw = image_batch(raw_images[:4]).reshape(4, 3, 1024)
    weights = rng.normal(0, 0.1, (12, 1, 5)).astype(np.float32)
    projection = rng.normal(0, 0.1, (7, 12)).astype(np.float32)
    a, w, p = [
        ts.tensor(value, device=device, requires_grad=True) for value in (raw, weights, projection)
    ]
    x, tw, tp = [
        torch.tensor(value, device=target, requires_grad=True)
        for value in (raw, weights, projection)
    ]
    actual = nn.functional.conv1d(a, w, stride=2, padding=4, dilation=2, groups=3).sigmoid().tanh()
    actual = nn.functional.avg_pool1d(nn.functional.max_pool1d(actual, 3, 2, 1), 3, 2, 1)
    actual = nn.functional.linear(actual.mean(-1), p)
    expected = F.conv1d(x, tw, stride=2, padding=4, dilation=2, groups=3).sigmoid().tanh()
    expected = F.avg_pool1d(F.max_pool1d(expected, 3, 2, 1), 3, 2, 1)
    expected = F.linear(expected.mean(-1), tp)
    loss, eloss = (actual**2).mean(), (expected**2).mean()
    loss.backward()
    eloss.backward()
    result = {
        "output": discrepancy(
            actual.numpy(),
            expected.detach().cpu().numpy(),
            atol=2e-5,
            rtol=2e-4,
            label="mixed op graph",
        )
    }
    for name, left, right in zip(["input", "conv_weight", "projection"], [a, w, p], [x, tw, tp]):
        result[name] = discrepancy(
            left.grad.numpy(),
            right.grad.cpu().numpy(),
            atol=2e-6,
            rtol=2e-3,
            label=f"mixed gradient {name}",
        )
    return result


def real_dataloader(saved, device, split):
    from tensorsmith.data import DataLoader, TensorDataset

    model = clone_model(saved, "cifar10", device).eval()
    raw, labels = split
    raw, labels = image_batch(raw[:129]), labels[:129]
    dataset = TensorDataset(
        ts.tensor(raw, device=device),
        ts.tensor(labels, device=device),
        ts.tensor(np.arange(129), device=device),
    )
    seen, total_loss = [], 0.0
    with ts.no_grad():
        for x, y, indices in DataLoader(dataset, batch_size=16, shuffle=True, seed=94):
            logits = model(x)
            total_loss += nn.functional.cross_entropy(logits, y, axis=-1).item() * len(y)
            seen.extend(indices.numpy().tolist())
        direct = nn.functional.cross_entropy(
            model(dataset.tensors[0]), dataset.tensors[1], axis=-1
        ).item()
    np.testing.assert_array_equal(np.sort(seen), np.arange(129))
    if abs(total_loss / 129 - direct) > 3e-5:
        raise AssertionError("DataLoader reordered/collated inputs or lost the short final batch")
    return {
        "real_images": 129,
        "batch_size": 16,
        "all_indices_seen_once": True,
        "short_final_batch": 1,
        "nll_error": abs(total_loss / 129 - direct),
    }


def low_precision_batch_norm(device, images):
    raw = image_batch(images[:64]).astype(np.float16)
    target = np.roll(raw, 1, axis=0)
    low = nn.BatchNorm2d(3, momentum=1.0).to(device, dtype="float16")
    reference = nn.BatchNorm2d(3, momentum=1.0).to(device)
    x = ts.tensor(raw, device=device, requires_grad=True)
    rx = ts.tensor(raw.astype(np.float32), device=device, requires_grad=True)
    actual = low(x)
    expected = reference(rx)
    ((actual - ts.tensor(target, device=device)) ** 2).mean().backward()
    ((expected - ts.tensor(target.astype(np.float32), device=device)) ** 2).mean().backward()
    result = {
        "input_shape": list(raw.shape),
        "statistics_values_per_channel": 65536,
        "output": discrepancy(
            actual.numpy(),
            expected.numpy(),
            atol=3e-3,
            rtol=3e-3,
            label="real CIFAR FP16 BatchNorm",
        ),
    }
    for name, a, b in [
        ("input_gradient", x.grad, rx.grad),
        ("gamma_gradient", low.weight.grad, reference.weight.grad),
        ("beta_gradient", low.bias.grad, reference.bias.grad),
        ("running_variance", low.running_var, reference.running_var),
    ]:
        result[name] = discrepancy(
            a.numpy(),
            b.numpy(),
            atol=1e-6 if name == "input_gradient" else 3e-3,
            rtol=5e-3,
            label=f"real FP16 BN {name}",
        )
    return result


def directional_derivatives(saved, task, batch):
    """Whole-parameter FP64 central differences, independent of any autodiff oracle."""
    model = clone_model(saved, task, "cpu", dtype="float64").eval()
    x, y = batch
    # Real inputs, full trained model; reduce only minibatch size for this
    # expensive numerical check, not model depth/width/vocabulary.
    x, y = x[:2], y[:2]
    inputs = ts.tensor(x, dtype="float64" if task == "cifar10" else "int64")
    labels = ts.tensor(y)
    loss = nn.functional.cross_entropy(model(inputs), labels, axis=-1)
    loss.backward()
    parameters = list(model.parameters())
    original = [p._data.copy() for p in parameters]
    derivatives = [p._grad.copy() for p in parameters]
    rng = np.random.default_rng(173)
    results = []
    for index in range(3):
        directions = [rng.standard_normal(p.shape) for p in parameters]
        norm = math.sqrt(sum(float((d * d).sum()) for d in directions))
        directions = [d / norm for d in directions]
        analytic = sum(float((g * d).sum()) for g, d in zip(derivatives, directions))
        measurements = []
        for epsilon in (1e-3, 3e-4, 1e-4):
            values = []
            for sign in (1, -1):
                for p, base, direction in zip(parameters, original, directions):
                    p._data = base + sign * epsilon * direction
                with ts.no_grad():
                    values.append(
                        nn.functional.cross_entropy(model(inputs), labels, axis=-1).item()
                    )
            numerical = (values[0] - values[1]) / (2 * epsilon)
            error = abs(numerical - analytic)
            measurements.append(
                {
                    "epsilon": epsilon,
                    "central_difference": numerical,
                    "absolute_error": error,
                    "relative_error": error / max(abs(analytic), 1e-10),
                }
            )
        best = min(measurements, key=lambda entry: entry["absolute_error"])
        if best["absolute_error"] > 1e-6 + abs(analytic) * 0.002:
            raise AssertionError(
                f"{task} whole-model directional gradient mismatch: {measurements}"
            )
        results.append({"direction": index, "analytic": analytic, "measurements": measurements})
    for p, base in zip(parameters, original):
        p._data = base
    return {
        "device": "cpu",
        "dtype": "float64",
        "model_parameter_elements": sum(p.size for p in parameters),
        "real_input_shape": list(x.shape),
        "directions": results,
    }


def forbid_bulk_host_fallback(saved, task, batch):
    """Fail if Metal indexed-update compatibility paths move float work to CPU."""
    # tensorsmith.device is also a public function: import the module explicitly.
    import importlib
    from unittest.mock import patch

    device_module = importlib.import_module("tensorsmith.device")
    model = clone_model(saved, task, "metal")
    original = device_module.asnumpy

    def guarded(value):
        if type(value).__module__.startswith("mlx") and "float" in str(value.dtype):
            raise AssertionError("floating GPU operator attempted host compatibility fallback")
        return original(value)

    x, y = batch
    with patch.object(device_module, "asnumpy", guarded):
        loss = nn.functional.cross_entropy(
            model(ts.tensor(x, device="metal")), ts.tensor(y, device="metal"), axis=-1
        )
        loss.backward()
        complete(model, values=[loss])
    return {
        "task": task,
        "input_shape": list(x.shape),
        "loss": loss.item(),
        "bulk_floating_cpu_fallbacks": 0,
        "note": "Explicit input transfer/label validation/output reporting are allowed; indexed-update floating CPU compatibility fallback is forbidden",
    }


def inference_memory_lifetime(saved, batch):
    """Completed full-vocabulary real-text logits must not accumulate with GC off."""
    from tensorsmith.device import xp_for

    model = clone_model(saved, "wikitext2", "metal")
    model.eval()
    mx = xp_for("metal")
    inputs = ts.tensor(batch[0], device="metal")
    with ts.no_grad():
        for _ in range(2):
            output = model(inputs)
            complete(model, values=[output])
            del output
        gc.collect()
        baseline = mx.get_active_memory()
        active = []
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(32):
                output = model(inputs)
                complete(model, values=[output])
                del output
                mx.synchronize()
                active.append(mx.get_active_memory())
        finally:
            if was_enabled:
                gc.enable()
        output_bytes = int(np.prod(batch[0].shape)) * model.config.vocab_size * 4
        require_growth = max(active) - baseline
        if require_growth > output_bytes:
            raise AssertionError(
                f"completed logits retained with cyclic GC disabled: growth={require_growth}, one_output={output_bytes}"
            )
    return {
        "completed_full_logits_forwards": 32,
        "cyclic_gc_disabled": True,
        "baseline_mlx_active_bytes": baseline,
        "samples_mlx_active_bytes": active,
        "maximum_active_growth_bytes": require_growth,
        "allowed_growth_bytes_one_output": output_bytes,
        "note": "Allocator cache is intentionally excluded; completed output arrays must not remain active",
    }


def main(args):
    language = ts.load(args.language_checkpoint)
    vision = ts.load(args.vision_checkpoint)
    tokens, _, _ = load_wikitext(args.cache)
    images, _ = load_cifar(args.cache)
    language_data = list(language_batches(tokens["valid"][: 4 * 513], 4, 128))[:4]
    vision_data = [
        (image_batch(images["valid"][0][i : i + 4]), images["valid"][1][i : i + 4])
        for i in (0, 4, 8)
    ]
    payload = {
        "source_sha256": source_digest(),
        "device": args.device,
        "checks": [],
        "language_checkpoint": str(args.language_checkpoint),
        "vision_checkpoint": str(args.vision_checkpoint),
        "acceptance_note": "Predeclared numerical thresholds; all gradient elements checked. These finite checks do not prove every supported configuration.",
        "metal_bulk_host_fallback_guard": args.device == "metal",
    }
    if args.device == "metal":
        import importlib
        from unittest.mock import patch

        device_module = importlib.import_module("tensorsmith.device")
        original_asnumpy = device_module.asnumpy

        def forbid_fallback(value):
            if type(value).__module__.startswith("mlx") and (
                "float" in str(value.dtype) or value.size > 4096
            ):
                raise AssertionError(
                    "native device helper attempted a bulk GPU-to-CPU compatibility fallback"
                )
            return original_asnumpy(value)

    def check(name, function):
        started = time.perf_counter()
        print(f"CHECK {name}", flush=True)
        try:
            if args.device == "metal":
                with patch.object(device_module, "asnumpy", forbid_fallback):
                    result = function()
            else:
                result = function()
            entry = {
                "name": name,
                "status": "pass",
                "seconds": time.perf_counter() - started,
                "result": result,
            }
        except Exception as error:  # noqa: BLE001 - record each independent acceptance failure
            entry = {
                "name": name,
                "status": "fail",
                "seconds": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        payload["checks"].append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"{entry['status'].upper()} {name} ({entry['seconds']:.1f}s) {entry.get('error', '')}",
            flush=True,
        )
        gc.collect()

    check(
        "complete_resnet_outputs_all_gradients_sgd_running_stats",
        lambda: oracle(vision, "cifar10", args.device, vision_data),
    )
    check(
        "complete_transformer_outputs_all_gradients_adamw",
        lambda: oracle(language, "wikitext2", args.device, language_data[:3]),
    )
    if args.device == "metal":
        check(
            "resnet_metal_backward_no_bulk_host_fallback",
            lambda: forbid_bulk_host_fallback(vision, "cifar10", vision_data[0]),
        )
        check(
            "transformer_metal_backward_no_bulk_host_fallback",
            lambda: forbid_bulk_host_fallback(language, "wikitext2", language_data[0]),
        )
        check(
            "real_language_completed_inference_releases_gpu_outputs",
            lambda: inference_memory_lifetime(language, language_data[0]),
        )
    check(
        "real_language_checkpointing_and_microbatch_accumulation",
        lambda: checkpoint_accumulation(language, args.device, language_data[0]),
    )
    check(
        "real_language_auto_dense_streaming_attention_gradients",
        lambda: attention_paths(language, args.device, language_data[0]),
    )
    check(
        "real_language_stochastic_checkpoint_dropout_rng",
        lambda: stochastic_checkpoint(language, args.device, language_data[0]),
    )
    if args.long_context:
        long_x = tokens["valid"][:2048].reshape(1, -1)
        long_y = tokens["valid"][1:2049].reshape(1, -1)
        check(
            "2048_real_token_full_transformer_all_gradients_oracle",
            lambda: oracle(language, "wikitext2", args.device, [(long_x, long_y)]),
        )
    if args.large_checkpoint:
        large = ts.load(args.large_checkpoint)
        real_x = tokens["valid"][:512].reshape(1, -1)
        real_y = tokens["valid"][1:513].reshape(1, -1)
        check(
            "129m_real_transformer_outputs_all_parameter_gradients_adamw",
            lambda: oracle(large, "wikitext2", args.device, [(real_x, real_y)]),
        )
    check(
        "resnet_model_optimizer_scheduler_save_resume",
        lambda: resume(vision, "cifar10", args.device, vision_data),
    )
    check(
        "transformer_model_optimizer_scheduler_save_resume",
        lambda: resume(language, "wikitext2", args.device, language_data[:3]),
    )
    check(
        "640_token_chunked_cache_int8_beams_greedy_generation",
        lambda: cached_language(language, args.device, tokens["valid"]),
    )
    check(
        "fp16_language_inference_and_scaled_training",
        lambda: low_precision(language, args.device, language_data),
    )
    check(
        "grouped_dilated_conv1d_sigmoid_tanh_pooling_composition",
        lambda: mixed_operator_graph(args.device, images["valid"][0]),
    )
    from extra_graph import verify as extra_graph

    check(
        "real_cifar_2d_pools_grouped_conv_bias_gelu_layernorm_all_gradients",
        lambda: extra_graph(args.device, images["valid"][0], images["valid"][1], discrepancy),
    )
    check(
        "real_cifar_dataloader_shuffle_collation_short_tail",
        lambda: real_dataloader(vision, args.device, images["valid"]),
    )
    check(
        "real_cifar_65536_value_fp16_batchnorm_reduction_gradients",
        lambda: low_precision_batch_norm(args.device, images["valid"][0]),
    )
    check(
        "fp64_resnet_whole_model_directional_derivatives_cpu",
        lambda: directional_derivatives(vision, "cifar10", vision_data[0]),
    )
    check(
        "fp64_transformer_whole_model_directional_derivatives_cpu",
        lambda: directional_derivatives(language, "wikitext2", language_data[0]),
    )
    payload["passed"] = sum(c["status"] == "pass" for c in payload["checks"])
    payload["failed"] = sum(c["status"] == "fail" for c in payload["checks"])
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if payload["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "metal", "cuda"], default="cpu")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--language-checkpoint", type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--large-checkpoint", type=Path)
    parser.add_argument(
        "--long-context",
        action="store_true",
        help="Also independently check a full 2048-token backward pass (context capacity extended only for this functional stress test)",
    )
    main(parser.parse_args())
