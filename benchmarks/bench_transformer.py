"""Attention, optimizer-inclusive training, prefill and cached/full-prefix decode.

Decode cases have identical fixed prefix/chunk lengths, not a changing cache.
No network/data downloads; pass --json for reproducible raw samples.
"""

import argparse

from common import add_timing_arguments, complete, devices, measure, report
from reference import torch_backend

import tensorsmith as ts
from tensorsmith import nn


def benchmark(device, args):
    if min(args.batch, args.seq_len, args.dim, args.heads, args.layers, args.vocab) <= 0:
        raise ValueError("model/workload dimensions must be positive")
    ts.seed(42)
    results = []
    dim = args.dim // args.heads
    qkv = [
        ts.randn(args.batch, args.heads, args.seq_len, dim, device=device, requires_grad=True)
        for _ in range(3)
    ]
    complete(device, qkv)
    for backend in (
        ["dense", "streaming", "native"] if device.type == "metal" else ["dense", "streaming"]
    ):
        with ts.no_grad():
            results.append(
                measure(
                    f"attention_{backend}_inference",
                    device,
                    lambda backend=backend: nn.scaled_dot_product_attention(
                        *qkv, is_causal=True, backend=backend
                    ),
                    args.iterations,
                    args.warmup,
                )
            )

        def attention(backend=backend):
            for tensor in qkv:
                tensor.zero_grad()
            loss = (
                nn.scaled_dot_product_attention(*qkv, is_causal=True, backend=backend) ** 2
            ).mean()
            loss.backward()
            return loss, [t.grad for t in qkv]

        results.append(
            measure(
                f"attention_{backend}_forward_backward",
                device,
                attention,
                args.iterations,
                args.warmup,
            )
        )

    if args.torch:
        torch, selected, torch_complete = torch_backend(device)
        tensors = [torch.tensor(t.numpy().copy(), device=selected, requires_grad=True) for t in qkv]

        def torch_attention():
            for t in tensors:
                t.grad = None
            output = torch.nn.functional.scaled_dot_product_attention(*tensors, is_causal=True)
            loss = (output**2).mean()
            loss.backward()
            return loss, [t.grad for t in tensors]

        import numpy as np

        for t in qkv:
            t.zero_grad()
        ours = nn.scaled_dot_product_attention(*qkv, is_causal=True)
        (ours**2).mean().backward()
        torch_attention()
        for t, reference in zip(qkv, tensors):
            np.testing.assert_allclose(
                t.grad.numpy(), reference.grad.detach().cpu().numpy(), rtol=5e-4, atol=5e-5
            )
        with torch.no_grad():
            results.append(
                measure(
                    "torch_attention_inference",
                    device,
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        *tensors, is_causal=True
                    ),
                    args.iterations,
                    args.warmup,
                    completion=torch_complete,
                )
            )
        results.append(
            measure(
                "torch_attention_forward_backward",
                device,
                torch_attention,
                args.iterations,
                args.warmup,
                completion=torch_complete,
            )
        )

    config = nn.TransformerConfig(
        args.vocab,
        dim=args.dim,
        num_heads=args.heads,
        num_layers=args.layers,
        max_seq_len=args.seq_len + 1,
        num_kv_heads=args.kv_heads,
        attention_backend=args.attention_backend,
        activation_checkpointing=args.activation_checkpointing,
    )
    model = nn.TransformerLM(config, device=device)
    tokens = ts.tensor([[i % args.vocab for i in range(args.seq_len)]] * args.batch, device=device)
    targets = ts.tensor(
        [[(i + 1) % args.vocab for i in range(args.seq_len)]] * args.batch, device=device
    )
    optimizer = ts.optim.AdamW(model.parameters(), lr=1e-3)
    parameters = list(model.parameters())
    complete(device, parameters)

    def training():
        optimizer.zero_grad()
        loss = model.loss(tokens, targets)
        loss.backward()
        optimizer.step()
        return loss, [p.grad for p in parameters], parameters, optimizer.state

    results.append(
        measure(
            "lm_training_step_including_adamw",
            device,
            training,
            args.iterations,
            args.warmup,
            args.batch * args.seq_len,
        )
    )
    model.eval()
    with ts.no_grad():
        caches = model.new_cache(args.batch, quantized=args.quantized_cache)

        def prefill():
            for cache in caches:
                cache.reset()
            output = model(tokens, caches=caches, logits_to_keep=1)
            return output, [c.storage for c in caches]

        results.append(
            measure(
                "lm_prefill_with_cache",
                device,
                prefill,
                args.iterations,
                args.warmup,
                args.batch * args.seq_len,
            )
        )
        complete(device, prefill())
        one_token = tokens[:, -1:]
        full_tokens = ts.cat((tokens, one_token), dim=1)

        def cached_decode():
            for cache in caches:
                cache.truncate(args.seq_len)
            return model(one_token, caches=caches, logits_to_keep=1), [c.storage for c in caches]

        def full_decode():
            return model(full_tokens, logits_to_keep=1)

        # An untimed correctness guard makes the speedup meaningful.
        import numpy as np

        actual, expected = cached_decode()[0].numpy(), full_decode().numpy()
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.05 if args.quantized_cache else 5e-4,
            atol=0.02 if args.quantized_cache else 5e-4,
        )
        results.append(
            measure(
                "lm_decode_one_token_cached",
                device,
                cached_decode,
                args.iterations,
                args.warmup,
                args.batch,
            )
        )
        results[-1]["kv_cache_bytes"] = sum(c.memory_bytes for c in caches)
        results[-1]["kv_cache_quantized"] = args.quantized_cache
        results[-1]["correctness_max_abs_logit_error"] = float(np.max(np.abs(actual - expected)))
        results.append(
            measure(
                "lm_decode_one_token_full_prefix",
                device,
                full_decode,
                args.iterations,
                args.warmup,
                args.batch,
            )
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_timing_arguments(parser)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument(
        "--quantized-cache",
        action="store_true",
        help="approximate int8 KV storage, not a fused int8 kernel",
    )
    parser.add_argument(
        "--torch",
        action="store_true",
        help="optional same-shape PyTorch SDPA baseline, not a different LM",
    )
    parser.add_argument(
        "--attention-backend", choices=["auto", "dense", "streaming", "native"], default="auto"
    )
    args = parser.parse_args()
    report([r for d in devices(args) for r in benchmark(d, args)], args)
