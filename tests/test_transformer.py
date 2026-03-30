"""Numerical VJP checks, accelerator parity, and cached-decoding contracts."""

import unittest


from unittest.mock import patch


import numpy as np


import tensorsmith as ts


from tensorsmith import nn


from tensorsmith.nn import functional as F


def check_vjp(test, function, arrays, checks=8):
    tensors = [ts.tensor(a.copy(), dtype="float64", requires_grad=True) for a in arrays]
    output = function(*tensors)
    upstream = np.random.default_rng(12).normal(size=output.shape)
    (output * ts.tensor(upstream, dtype="float64")).sum().backward()
    for index, raw in enumerate(arrays):
        for flat in np.linspace(0, raw.size - 1, min(checks, raw.size), dtype=int):
            position = np.unravel_index(flat, raw.shape)
            original = raw[position]
            raw[position] = original + 1e-5
            high = (
                function(*[ts.tensor(a, dtype="float64") for a in arrays]).numpy() * upstream
            ).sum()
            raw[position] = original - 1e-5
            low = (
                function(*[ts.tensor(a, dtype="float64") for a in arrays]).numpy() * upstream
            ).sum()
            raw[position] = original
            test.assertAlmostEqual(
                tensors[index].grad.numpy()[position], (high - low) / 2e-5, places=5
            )


class TransformerTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(19)

    def test_attention_dense_and_streaming_vjp(self):
        for backend in ("dense", "streaming"):
            with self.subTest(backend=backend):
                arrays = [
                    np.random.randn(1, 2, 3, 4),
                    np.random.randn(1, 2, 5, 4),
                    np.random.randn(1, 2, 5, 3),
                ]
                check_vjp(
                    self,
                    lambda q, k, v, backend=backend: nn.scaled_dot_product_attention(
                        q, k, v, is_causal=True, causal_offset=2, backend=backend, block_size=2
                    ),
                    arrays,
                )

    def test_masks_all_masked_and_backend_parity(self):
        raw = [np.random.randn(2, 2, 4, 4).astype(np.float32) for _ in range(3)]
        boolean = np.ones((4, 4), dtype=bool)
        boolean[1] = False
        boolean[3, 0] = False
        expected = None
        for device in ts.available_devices():
            backends = (
                ("dense", "streaming", "native")
                if device.type == "metal"
                else ("dense", "streaming")
            )
            for backend in backends:
                for mask in (boolean, np.where(boolean, 0, -np.inf).astype(np.float32)):
                    with self.subTest(
                        device=str(device), backend=backend, bool_mask=mask.dtype == bool
                    ):
                        q, k, v = [ts.tensor(a, device=device, requires_grad=True) for a in raw]
                        output = nn.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            ts.tensor(mask, device=device),
                            is_causal=True,
                            backend=backend,
                            block_size=2,
                        )
                        (output**2).sum().backward()
                        current = [output.numpy(), q.grad.numpy(), k.grad.numpy(), v.grad.numpy()]
                        self.assertTrue(all(np.isfinite(a).all() for a in current))
                        np.testing.assert_array_equal(current[0][:, :, 1], 0)
                        np.testing.assert_array_equal(current[1][:, :, 1], 0)
                        if expected is None:
                            expected = current
                        else:
                            for a, b in zip(current, expected):
                                np.testing.assert_allclose(a, b, rtol=1e-4, atol=3e-5)

    def test_grouped_attention_vjp_and_native_parity(self):
        raw = [
            np.random.randn(1, 4, 3, 4),
            np.random.randn(1, 2, 5, 4),
            np.random.randn(1, 2, 5, 4),
        ]
        check_vjp(
            self,
            lambda q, k, v: nn.scaled_dot_product_attention(
                q, k, v, enable_gqa=True, backend="streaming", block_size=2
            ),
            raw,
        )
        expected = None
        for device in ts.available_devices():
            tensors = [
                ts.tensor(a.astype(np.float32), device=device, requires_grad=True) for a in raw
            ]
            output = nn.scaled_dot_product_attention(*tensors, enable_gqa=True)
            (output**2).sum().backward()
            current = [output.numpy()] + [t.grad.numpy() for t in tensors]
            if expected is None:
                expected = current
            else:
                for a, b in zip(current, expected):
                    np.testing.assert_allclose(a, b, rtol=2e-4, atol=4e-5)

    def test_linear_batched_and_vector_vjp(self):
        for shape in ((4,), (2, 3, 4)):
            check_vjp(
                self, F.linear, [np.random.randn(*shape), np.random.randn(5, 4), np.random.randn(5)]
            )

    def test_native_unmasked_attention_gradient_parity(self):
        raw = [np.random.randn(1, 2, 7, 8).astype(np.float32) for _ in range(3)]
        for causal in (False, True):
            expected = None
            for device in ts.available_devices():
                tensors = [ts.tensor(a, device=device, requires_grad=True) for a in raw]
                output = nn.scaled_dot_product_attention(*tensors, is_causal=causal)
                (output**2).sum().backward()
                current = [output.numpy()] + [t.grad.numpy() for t in tensors]
                if expected is None:
                    expected = current
                else:
                    for a, b in zip(current, expected):
                        np.testing.assert_allclose(a, b, rtol=1e-4, atol=3e-5)

    def test_native_inference_skips_backward_only_graphs(self):
        if not ts.is_available("metal"):
            self.skipTest("Metal unavailable")
        from tensorsmith.device import xp_for

        mx = xp_for("metal")
        q = ts.randn(1, 4, 3, 8, device="metal", requires_grad=True)
        k = ts.randn(1, 2, 5, 8, device="metal", requires_grad=True)
        v = ts.randn(1, 2, 5, 8, device="metal", requires_grad=True)
        norm = nn.RMSNorm(8, device="metal")
        with (
            ts.no_grad(),
            patch.object(mx, "matmul", side_effect=AssertionError("VJP-only matmul")),
            patch.object(mx, "broadcast_to", side_effect=AssertionError("KV expansion")),
            patch.object(mx, "mean", side_effect=AssertionError("VJP-only norm statistics")),
        ):
            output = nn.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            normalized = norm(q)
            ts.evaluate(output, normalized)
        self.assertTrue(np.isfinite(output.numpy()).all())
        self.assertTrue(np.isfinite(normalized.numpy()).all())

    def test_dropout_attention_gradient(self):
        q, k, v = [ts.randn(1, 1, 3, 4, requires_grad=True) for _ in range(3)]
        ts.seed(8)
        output = nn.scaled_dot_product_attention(q, k, v, dropout_p=0.3)
        output.sum().backward()
        self.assertTrue(all(np.isfinite(t.grad.numpy()).all() for t in (q, k, v)))
        with self.assertRaises(ValueError):
            nn.scaled_dot_product_attention(q, k, v, dropout_p=0.1, backend="streaming")

    def test_rope_partial_offset_vjp(self):
        raw = np.random.randn(2, 3, 6)
        check_vjp(self, lambda x: nn.rotary_embedding(x, offset=5, rotary_dim=4), [raw])
        for device in ts.available_devices():
            x = ts.tensor(raw.astype(np.float32), device=device)
            rotated = nn.rotary_embedding(x, 5, rotary_dim=4).numpy()
            np.testing.assert_allclose((rotated**2).sum(-1), (raw**2).sum(-1), rtol=2e-6)
            np.testing.assert_array_equal(rotated[..., 4:], x.numpy()[..., 4:])

    def test_norms_and_activation_vjps(self):
        raw = np.random.randn(2, 3, 4)
        weight = np.random.randn(4)
        bias = np.random.randn(4)
        residual = np.random.randn(*raw.shape)
        check_vjp(self, F.silu, [raw.copy()])
        check_vjp(self, F.bias_gelu, [raw.copy(), bias.copy()])
        check_vjp(self, F.layer_norm, [raw.copy(), weight.copy(), bias.copy()])
        check_vjp(self, F.rms_norm, [raw.copy(), weight.copy()])
        check_vjp(
            self, F.residual_layer_norm, [raw.copy(), residual.copy(), weight.copy(), bias.copy()]
        )
        check_vjp(self, F.residual_rms_norm, [raw.copy(), residual.copy(), weight.copy()])

    def test_norm_accelerator_and_float16_parity(self):
        raw = [np.random.randn(2, 3, 8).astype(np.float32), np.random.randn(8).astype(np.float32)]
        for dtype in ("float32", "float16"):
            for function in (F.rms_norm, F.layer_norm):
                expected = None
                for device in ts.available_devices():
                    tensors = [
                        ts.tensor(a, device=device, dtype=dtype, requires_grad=True) for a in raw
                    ]
                    output = function(*tensors)
                    (output**2).sum().backward()
                    current = [output.numpy()] + [t.grad.numpy() for t in tensors]
                    if expected is None:
                        expected = current
                    else:
                        for a, b in zip(current, expected):
                            np.testing.assert_allclose(
                                a,
                                b,
                                rtol=0.02 if dtype == "float16" else 1e-4,
                                atol=0.02 if dtype == "float16" else 2e-5,
                            )

    def test_cross_entropy_large_vocab_without_eye(self):
        for device in ts.available_devices():
            with patch("numpy.eye", side_effect=AssertionError("quadratic vocabulary allocation")):
                logits = ts.zeros(2, 3, 50000, device=device, requires_grad=True)
                targets = ts.tensor([[0, 49999, -100], [123, 2, 49]], device=device)
                loss = F.cross_entropy(logits, targets, axis=-1)
                loss.backward()
                self.assertAlmostEqual(loss.item(), np.log(50000), places=5)
                np.testing.assert_array_equal(logits.grad.numpy()[0, 2], 0)
                self.assertEqual(logits.grad.shape, logits.shape)

    def test_cross_entropy_smoothing_ignore_vjp(self):
        labels = ts.tensor([[0, -100, 3], [2, 1, 0]])
        raw = np.random.randn(2, 3, 4)
        check_vjp(
            self,
            lambda x: F.cross_entropy(x, labels, axis=-1, label_smoothing=0.2, reduction="none"),
            [raw],
        )
        for device in ts.available_devices():
            logits = ts.tensor(raw.astype(np.float32), device=device, requires_grad=True)
            loss = F.cross_entropy(logits, ts.tensor(np.full((2, 3), -100), device=device), axis=-1)
            loss.backward()
            self.assertEqual(loss.item(), 0)
            np.testing.assert_array_equal(logits.grad.numpy(), 0)
            with self.assertRaises(ValueError):
                F.cross_entropy(logits, ts.tensor(np.full((2, 3), 4), device=device), axis=-1)

    def test_cached_logits_match_full_prefix(self):
        for device in ts.available_devices():
            for backend in ("auto", "streaming"):
                with self.subTest(device=str(device), backend=backend):
                    model = nn.TransformerLM(
                        nn.TransformerConfig(
                            17,
                            dim=16,
                            num_heads=4,
                            num_kv_heads=2,
                            num_layers=2,
                            max_seq_len=16,
                            attention_backend=backend,
                        ),
                        device=device,
                    ).eval()
                    tokens = ts.tensor([[1, 2, 3, 4, 5, 6], [4, 3, 2, 1, 0, 7]], device=device)
                    with ts.no_grad():
                        expected = model(tokens).numpy()
                        caches = model.new_cache(2)
                        chunks = [
                            model(tokens[:, :3], caches=caches),
                            model(tokens[:, 3:5], caches=caches),
                            model(tokens[:, 5:], caches=caches),
                        ]
                        ts.evaluate(chunks, [c.storage for c in caches])
                        np.testing.assert_allclose(
                            ts.cat(chunks, dim=1).numpy(), expected, rtol=3e-4, atol=3e-5
                        )
                        self.assertTrue(all(c.length == 6 for c in caches))
                        quantized = model.new_cache(2, quantized=True)
                        actual = model(tokens, caches=quantized).numpy()
                        np.testing.assert_allclose(actual, expected, rtol=0.08, atol=0.006)

    def test_causal_no_future_leakage_and_gradients(self):
        model = nn.TransformerLM(
            nn.TransformerConfig(
                11, dim=16, num_heads=4, num_kv_heads=1, num_layers=2, max_seq_len=8
            )
        )
        tokens = ts.tensor([[1, 2, 3, 4]])
        logits = model(tokens)
        altered = model(ts.tensor([[1, 2, 9, 8]]))
        np.testing.assert_allclose(logits.numpy()[:, :2], altered.numpy()[:, :2], atol=1e-6)
        loss = F.cross_entropy(logits, ts.tensor([[2, 3, 4, 5]]), axis=-1)
        loss.backward()
        parameters = list(model.parameters())
        self.assertTrue(
            all(p.grad is not None and np.isfinite(p.grad.numpy()).all() for p in parameters)
        )
        self.assertEqual(sum(p is model.token_embedding.weight for p in parameters), 1)
        clone = nn.TransformerLM(model.config)
        clone.load_state_dict(model.state_dict())
        np.testing.assert_allclose(clone(tokens).numpy(), logits.numpy(), atol=1e-6)
        np.testing.assert_allclose(
            model(tokens, logits_to_keep=1).numpy(), logits.numpy()[:, -1:], atol=1e-6
        )
        with self.assertRaises(ValueError):
            model(tokens, logits_to_keep=5)

    def test_toy_language_model_learns(self):
        model = nn.TransformerLM(
            nn.TransformerConfig(4, dim=16, num_heads=2, num_layers=1, hidden_dim=24, max_seq_len=8)
        )
        optimizer = ts.optim.AdamW(model.parameters(), lr=0.02)
        x = ts.tensor([[0, 1, 2, 3, 0, 1]])
        y = ts.tensor([[1, 2, 3, 0, 1, 2]])
        initial = model.loss(x, y).item()
        for _ in range(40):
            optimizer.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            nn.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        self.assertLess(model.loss(x, y).item(), initial * 0.15)
