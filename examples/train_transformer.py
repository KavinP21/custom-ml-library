"""Tiny character LM: custom autograd, GQA/RoPE, accumulation, and KV decoding.

Use --text FILE for your own UTF-8 corpus. The default is an offline smoke test,
not a meaningful language-model evaluation. Training checkpoints resume model,
optimizer, scheduler, scaler and NumPy batch-selection RNG state.
"""

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

import tensorsmith as ts
from tensorsmith import nn


def train(args):
    if min(args.steps, args.batch, args.seq_len, args.accumulate, args.log_every) <= 0:
        raise ValueError(
            "steps, batch, sequence length, accumulation and log interval must be positive"
        )
    ts.seed(42)
    text = (
        Path(args.text).read_text(encoding="utf-8")
        if args.text
        else (
            "tensor smith learns patterns. attention connects tokens. cache reuses history.\n" * 100
        )
    )
    chars = sorted(set(text))
    if not chars or len(text) <= args.seq_len + 1:
        raise ValueError("corpus must exceed sequence length + 1")
    char_to_id = {char: i for i, char in enumerate(chars)}
    encoded = np.array([char_to_id[c] for c in text], dtype=np.int32)
    corpus_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    config = nn.TransformerConfig(
        len(chars),
        dim=args.dim,
        num_heads=4,
        num_kv_heads=2,
        num_layers=args.layers,
        hidden_dim=3 * args.dim,
        max_seq_len=max(256, args.seq_len),
        dropout=0,
        activation_checkpointing=args.activation_checkpointing,
    )
    model = nn.TransformerLM(config, device=args.device).to(dtype=args.dtype)
    optimizer = ts.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ts.optim.WarmupCosineLR(
        optimizer, min(10, args.steps // 5), args.steps, min_lr=args.lr / 10
    )
    scaler = ts.amp.GradScaler(init_scale=128, enabled=args.dtype == "float16")
    first_step = 0
    if args.resume:
        checkpoint = ts.load(args.resume)
        if (
            checkpoint["config"] != asdict(config)
            or checkpoint["chars"] != chars
            or checkpoint["corpus_hash"] != corpus_hash
            or checkpoint["training_dtype"] != args.dtype
        ):
            raise ValueError("checkpoint configuration/vocabulary does not match")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        np.random.set_state(checkpoint["numpy_rng"])
        first_step = checkpoint["step"]
        if first_step > args.steps:
            raise ValueError("--steps must be at least the completed checkpoint step")
    parameters = list(model.parameters())
    for step in range(first_step, args.steps):
        optimizer.zero_grad()
        losses = []
        for _ in range(args.accumulate):
            starts = np.random.randint(0, len(encoded) - args.seq_len - 1, size=args.batch)
            batches = np.stack([encoded[start : start + args.seq_len + 1] for start in starts])
            inputs = ts.tensor(batches[:, :-1], device=args.device)
            targets = ts.tensor(batches[:, 1:], device=args.device)
            loss = model.loss(inputs, targets)
            scaler.scale(loss / args.accumulate).backward()
            losses.append(loss)
        finite = scaler.unscale_(optimizer)
        # Overflow must skip clipping too, because nonfinite norms are rejected.
        if finite:
            nn.clip_grad_norm_(parameters, 1.0)
        applied = scaler.step(optimizer)
        scaler.update()
        if applied:
            scheduler.step()
        ts.evaluate(losses, parameters, optimizer.state)
        if step % args.log_every == 0 or step == args.steps - 1:
            print(
                f"step={step + 1} loss={sum(value.item() for value in losses) / len(losses):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f} scale={scaler.get_scale():.1f}"
            )
    if args.checkpoint:
        ts.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": args.steps,
                "config": asdict(config),
                "chars": chars,
                "numpy_rng": np.random.get_state(),
                "corpus_hash": corpus_hash,
                "training_dtype": args.dtype,
            },
            args.checkpoint,
        )
    prompt = args.prompt or text[:16]
    if len(prompt) > config.max_seq_len or args.generate < 0:
        raise ValueError("invalid prompt/generation length")
    if any(c not in char_to_id for c in prompt):
        raise ValueError("prompt includes characters outside the training vocabulary")
    tokens = ts.tensor([[char_to_id[c] for c in prompt]], device=args.device)
    generated = model.generate(
        tokens,
        min(args.generate, config.max_seq_len - len(prompt)),
        temperature=args.temperature,
        top_k=min(10, len(chars)),
    )
    print("".join(chars[int(i)] for i in generated.numpy()[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "metal"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--text")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--generate", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    train(parser.parse_args())
