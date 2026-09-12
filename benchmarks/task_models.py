"""Nontrivial task architectures and independently differentiated references."""

import math

import numpy as np

import tensorsmith as ts
from tensorsmith import nn


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        # Projection shortcut rather than the original paper's zero-padding.
        self.shortcut = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))
            if cin != cout or stride != 1
            else None
        )

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        return (self.bn2(self.conv2(self.bn1(self.conv1(x)).relu())) + residual).relu()


class CifarResNet(nn.Module):
    """20-layer CIFAR residual CNN, three 16/32/64-channel stages."""

    def __init__(self, blocks_per_stage=3, width=16):
        super().__init__()
        self.conv = nn.Conv2d(3, width, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(width)
        self.blocks = []
        cin = width
        for stage in range(3):
            cout = width * 2**stage
            for i in range(blocks_per_stage):
                self.blocks.append(ResidualBlock(cin, cout, 2 if stage and i == 0 else 1))
                cin = cout
        self.fc = nn.Linear(cin, 10)
        # He fan-out initialization, including shortcut convolutions.
        for name, p in self.named_parameters():
            if p.ndim == 4:
                p._data = np.random.normal(
                    0, math.sqrt(2 / (p.shape[0] * p.shape[2] * p.shape[3])), p.shape
                ).astype(np.float32)

    def forward(self, x):
        x = self.bn(self.conv(x)).relu()
        for block in self.blocks:
            x = block(x)
        return self.fc(x.mean((2, 3)))


def build_model(task, args, vocabulary_size=None):
    ts.seed(args.seed)
    if task == "cifar10":
        return CifarResNet(args.blocks_per_stage, args.width)
    config = nn.TransformerConfig(
        vocabulary_size,
        dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        hidden_dim=args.hidden_dim,
        max_seq_len=max(1024, args.seq_len),
        dropout=args.dropout,
        activation_checkpointing=args.activation_checkpointing,
    )
    return nn.TransformerLM(config)


def torch_reference(task, model, device):
    """PyTorch is a benchmark-only dependency; architectures/weights are matched.

    Functional reference keeps the exact TensorSmith state names and uses
    native torch autograd, convolution, batch normalization and SDPA.
    """
    import torch
    from torch.nn import functional as F

    class Reference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.names = [name for name, _ in model.named_parameters()]
            self.values = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.tensor(p.numpy(), device=device))
                    for _, p in model.named_parameters()
                ]
            )
            self.weights = dict(zip(self.names, self.values))
            self.buffers_by_name = {}
            buffers = [
                (name, value)
                for name, value in model.state_dict().items()
                if name not in self.names
            ]
            for i, (name, value) in enumerate(buffers):
                self.register_buffer(f"buffer_{i}", torch.tensor(value, device=device))
                self.buffers_by_name[name] = getattr(self, f"buffer_{i}")

        def forward(self, x):
            if task == "wikitext2":
                config = model.config
                b, length = x.shape
                d = config.dim // config.num_heads
                kv = config.num_kv_heads or config.num_heads
                h = F.embedding(x, self.weights["token_embedding.weight"])
                positions = torch.arange(length, device=x.device, dtype=torch.float32)
                frequencies = 10000 ** (
                    -torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d
                )
                angles = positions[:, None] * frequencies
                cos = torch.repeat_interleave(angles.cos(), 2, -1).to(h.dtype)
                sin = torch.repeat_interleave(angles.sin(), 2, -1).to(h.dtype)

                def rms(t, name):
                    if hasattr(F, "rms_norm"):
                        return F.rms_norm(t, (config.dim,), self.weights[name], eps=1e-6)
                    return (
                        t
                        * torch.rsqrt((t.float() ** 2).mean(-1, keepdim=True) + 1e-6).to(t.dtype)
                        * self.weights[name]
                    )

                def rope(t):
                    rotated = torch.stack((-t[..., 1::2], t[..., 0::2]), -1).reshape(t.shape)
                    return t * cos + rotated * sin

                for i in range(config.num_layers):
                    prefix = f"blocks.{i}."
                    normalized = rms(h, prefix + "attention_norm.weight")

                    def proj(name, heads, normalized=normalized, prefix=prefix):
                        return (
                            F.linear(
                                normalized, self.weights[prefix + "attention." + name + ".weight"]
                            )
                            .reshape(b, length, heads, d)
                            .permute(0, 2, 1, 3)
                        )

                    q, k, v = (
                        rope(proj("q_proj", config.num_heads)),
                        rope(proj("k_proj", kv)),
                        proj("v_proj", kv),
                    )
                    # Explicit repeat matches GQA semantics on all torch devices.
                    k, v = (
                        k.repeat_interleave(config.num_heads // kv, 1),
                        v.repeat_interleave(config.num_heads // kv, 1),
                    )
                    a = (
                        F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            is_causal=True,
                            dropout_p=config.dropout if self.training else 0,
                        )
                        .permute(0, 2, 1, 3)
                        .reshape(b, length, config.dim)
                    )
                    h = h + F.dropout(
                        F.linear(a, self.weights[prefix + "attention.out_proj.weight"]),
                        config.dropout,
                        self.training,
                    )
                    normalized = rms(h, prefix + "ffn_norm.weight")
                    activated = F.silu(
                        F.linear(normalized, self.weights[prefix + "ffn.gate.weight"])
                    ) * F.linear(normalized, self.weights[prefix + "ffn.up.weight"])
                    h = h + F.dropout(
                        F.linear(activated, self.weights[prefix + "ffn.down.weight"]),
                        config.dropout,
                        self.training,
                    )
                return F.linear(rms(h, "final_norm.weight"), self.weights["token_embedding.weight"])

            def conv(t, name, stride=1, padding=1):
                return F.conv2d(t, self.weights[name + ".weight"], stride=stride, padding=padding)

            def bn(t, name):
                return F.batch_norm(
                    t,
                    self.buffers_by_name[name + ".running_mean"],
                    self.buffers_by_name[name + ".running_var"],
                    self.weights[name + ".weight"],
                    self.weights[name + ".bias"],
                    self.training,
                    0.1,
                    1e-5,
                )

            h = bn(conv(x, "conv"), "bn").relu()
            for i, block in enumerate(model.blocks):
                prefix = f"blocks.{i}"
                stride = block.conv1.stride
                residual = (
                    h
                    if block.shortcut is None
                    else bn(
                        conv(h, prefix + ".shortcut.layers.0", stride=stride, padding=0),
                        prefix + ".shortcut.layers.1",
                    )
                )
                h = (
                    bn(
                        conv(
                            bn(conv(h, prefix + ".conv1", stride=stride), prefix + ".bn1").relu(),
                            prefix + ".conv2",
                        ),
                        prefix + ".bn2",
                    )
                    + residual
                ).relu()
            return F.linear(h.mean((2, 3)), self.weights["fc.weight"], self.weights["fc.bias"])

    return Reference()
