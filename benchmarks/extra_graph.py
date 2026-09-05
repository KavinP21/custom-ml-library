"""Real-image classifier oracle for 2D pools, grouped conv and fused GELU/norm."""

import numpy as np
from task_data import image_batch

import tensorsmith as ts
from tensorsmith import nn


def verify(device, images, labels, compare):
    import torch
    from torch.nn import functional as F

    td = "mps" if device == "metal" else device
    rng = np.random.default_rng(98)
    raw = {
        "input": image_batch(images[:8]),
        "conv1_weight": rng.normal(0, 0.15, (16, 3, 3, 3)).astype(np.float32),
        "conv1_bias": rng.normal(0, 0.01, 16).astype(np.float32),
        "conv2_weight": rng.normal(0, 0.1, (32, 4, 3, 3)).astype(np.float32),
        "conv2_bias": rng.normal(0, 0.01, 32).astype(np.float32),
        "gelu_bias": rng.normal(0, 0.01, 32).astype(np.float32),
        "norm_weight": np.ones(2048, np.float32),
        "norm_bias": np.zeros(2048, np.float32),
        "linear_weight": rng.normal(0, 0.03, (10, 2048)).astype(np.float32),
        "linear_bias": np.zeros(10, np.float32),
    }
    a = {name: ts.tensor(value, device=device, requires_grad=True) for name, value in raw.items()}
    e = {name: torch.tensor(value, device=td, requires_grad=True) for name, value in raw.items()}
    out = nn.functional.conv2d(a["input"], a["conv1_weight"], a["conv1_bias"], padding=1).relu()
    out = nn.functional.max_pool2d(out, 2, 2)
    out = nn.functional.conv2d(
        out, a["conv2_weight"], a["conv2_bias"], padding=2, dilation=2, groups=4
    )
    out = nn.functional.bias_gelu(out.permute(0, 2, 3, 1), a["gelu_bias"]).permute(0, 3, 1, 2)
    out = nn.functional.avg_pool2d(out, 3, 2, 1).reshape(8, 2048)
    out = nn.functional.layer_norm(out, a["norm_weight"], a["norm_bias"])
    out = nn.functional.linear(out, a["linear_weight"], a["linear_bias"])
    expected = F.conv2d(e["input"], e["conv1_weight"], e["conv1_bias"], padding=1).relu()
    expected = F.max_pool2d(expected, 2, 2)
    expected = F.conv2d(
        expected, e["conv2_weight"], e["conv2_bias"], padding=2, dilation=2, groups=4
    )
    expected = F.gelu(expected + e["gelu_bias"].reshape(1, 32, 1, 1), approximate="tanh")
    expected = F.avg_pool2d(expected, 3, 2, 1).reshape(8, 2048)
    expected = F.layer_norm(expected, (2048,), e["norm_weight"], e["norm_bias"], eps=1e-5)
    expected = F.linear(expected, e["linear_weight"], e["linear_bias"])
    loss = nn.functional.cross_entropy(out, ts.tensor(labels[:8], device=device), axis=-1)
    reference_loss = F.cross_entropy(expected, torch.tensor(labels[:8], device=td))
    loss.backward()
    reference_loss.backward()
    result = {
        "real_image_shape": list(raw["input"].shape),
        "loss_error": abs(loss.item() - reference_loss.item()),
        "output": compare(
            out.numpy(),
            expected.detach().cpu().numpy(),
            atol=3e-4,
            rtol=2e-3,
            label="pooled/fused image classifier",
        ),
        "gradients": {},
    }
    if result["loss_error"] > 3e-5:
        raise AssertionError("pooled image classifier loss mismatch")
    for name in raw:
        result["gradients"][name] = compare(
            a[name].grad.numpy(),
            e[name].grad.cpu().numpy(),
            atol=3e-5,
            rtol=3e-3,
            label=f"pooled/fused graph {name}",
        )
    return result
