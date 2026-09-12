"""Optional independent oracle; PyTorch is NOT used by the framework."""

import importlib.util
import unittest

import numpy as np

import tensorsmith as ts
from tensorsmith import nn


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch oracle not installed")
class TorchReferenceTests(unittest.TestCase):
    def test_complete_gqa_rope_language_model_forward_and_backward(self):
        import torch
        from torch.nn import functional as TF

        for device in ts.available_devices():
            torch_device = "mps" if device.type == "metal" else str(device)
            if torch_device == "mps" and not torch.backends.mps.is_available():
                continue
            config = nn.TransformerConfig(
                19, dim=16, num_heads=4, num_kv_heads=2, num_layers=2, hidden_dim=24, max_seq_len=8
            )
            model = nn.TransformerLM(config, device=device)
            weights = {
                name: torch.tensor(p.numpy(), device=torch_device, requires_grad=True)
                for name, p in model.named_parameters()
            }
            raw_tokens = np.array([[1, 2, 3, 4], [4, 3, 1, 2]], dtype=np.int64)
            raw_labels = np.array([[2, 3, 4, 5], [3, 1, 2, 6]], dtype=np.int64)
            ids = torch.tensor(raw_tokens, device=torch_device)
            labels = torch.tensor(raw_labels, device=torch_device)
            hidden = TF.embedding(ids, weights["token_embedding.weight"])

            def rms(x, weight):
                return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * weight

            def rope(x, torch_device=torch_device):
                frequency = 10000 ** (
                    -torch.arange(0, 4, 2, device=torch_device, dtype=torch.float32) / 4
                )
                angle = (
                    torch.arange(4, device=torch_device, dtype=torch.float32)[:, None] * frequency
                )
                cosine = torch.repeat_interleave(torch.cos(angle), 2, dim=-1)
                sine = torch.repeat_interleave(torch.sin(angle), 2, dim=-1)
                rotated = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).reshape(x.shape)
                return x * cosine + rotated * sine

            for layer in range(2):
                prefix = f"blocks.{layer}."
                normalized = rms(hidden, weights[prefix + "attention_norm.weight"])

                def projection(name, heads, normalized=normalized, prefix=prefix, weights=weights):
                    return (
                        TF.linear(normalized, weights[prefix + "attention." + name + ".weight"])
                        .reshape(2, 4, heads, 4)
                        .permute(0, 2, 1, 3)
                    )

                q = rope(projection("q_proj", 4))
                k = rope(projection("k_proj", 2)).repeat_interleave(2, dim=1)
                v = projection("v_proj", 2).repeat_interleave(2, dim=1)
                attended = (
                    TF.scaled_dot_product_attention(q, k, v, is_causal=True)
                    .permute(0, 2, 1, 3)
                    .reshape(2, 4, 16)
                )
                hidden = hidden + TF.linear(attended, weights[prefix + "attention.out_proj.weight"])
                normalized = rms(hidden, weights[prefix + "ffn_norm.weight"])
                activated = TF.silu(
                    TF.linear(normalized, weights[prefix + "ffn.gate.weight"])
                ) * TF.linear(normalized, weights[prefix + "ffn.up.weight"])
                hidden = hidden + TF.linear(activated, weights[prefix + "ffn.down.weight"])
            expected = TF.linear(
                rms(hidden, weights["final_norm.weight"]), weights["token_embedding.weight"]
            )
            expected_loss = TF.cross_entropy(
                expected.reshape(-1, 19), labels.reshape(-1), label_smoothing=0.1
            )
            expected_loss.backward()
            actual = model(ts.tensor(raw_tokens, device=device))
            actual_loss = nn.functional.cross_entropy(
                actual, ts.tensor(raw_labels, device=device), axis=-1, label_smoothing=0.1
            )
            actual_loss.backward()
            np.testing.assert_allclose(
                actual.numpy(), expected.detach().cpu().numpy(), rtol=3e-4, atol=2e-5
            )
            self.assertAlmostEqual(actual_loss.item(), expected_loss.item(), places=5)
            for name, p in model.named_parameters():
                np.testing.assert_allclose(
                    p.grad.numpy(),
                    weights[name].grad.cpu().numpy(),
                    rtol=4e-4,
                    atol=3e-5,
                    err_msg=f"gradient mismatch: {name} on {device}",
                )
