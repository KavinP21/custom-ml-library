import tempfile
import unittest
from pathlib import Path

import numpy as np

import tensorsmith as ts
from tensorsmith import nn


class TrainingToolsTests(unittest.TestCase):
    def test_optimizer_resume_matches_uninterrupted(self):
        for optimizer_class in (ts.optim.AdamW, ts.optim.SGD):
            for device in ts.available_devices():
                parameter = nn.Parameter(ts.tensor([1.0, 2.0, 3.0], device=device))
                kwargs = {"lr": 0.01, "weight_decay": 0.02}
                kwargs.update(
                    {"amsgrad": True} if optimizer_class is ts.optim.AdamW else {"momentum": 0.9}
                )
                optimizer = optimizer_class([parameter], **kwargs)
                parameter.grad = ts.tensor([0.3, 0.2, 0.1], device=device)
                optimizer.step()
                state = optimizer.state_dict()
                clone = nn.Parameter(parameter.numpy().copy(), device=device)
                restored = optimizer_class([clone], lr=1.0)
                restored.load_state_dict(state)
                for _ in range(3):
                    parameter.grad = ts.tensor([0.1, 0.4, 0.2], device=device)
                    clone.grad = parameter.grad.clone()
                    optimizer.step()
                    restored.step()
                ts.evaluate(parameter, clone, optimizer.state, restored.state)
                np.testing.assert_allclose(parameter.numpy(), clone.numpy(), atol=1e-7)
                first_key = "exp_avg" if optimizer_class is ts.optim.AdamW else "momentum_buffer"
                self.assertFalse(
                    np.array_equal(
                        state["state"][0][first_key], optimizer.state_dict()["state"][0][first_key]
                    )
                )

    def test_fp16_adam_master_weights_preserve_small_updates(self):
        for device in ts.available_devices():
            parameter = nn.Parameter(ts.tensor([1.0], device=device, dtype="float16"))
            optimizer = ts.optim.AdamW([parameter], lr=1e-5)
            for _ in range(100):
                parameter.grad = ts.tensor([0.001], device=device, dtype="float16")
                optimizer.step()
                ts.evaluate(parameter, optimizer.state)
            self.assertLess(parameter.item(), 1.0)
            self.assertEqual(str(parameter.dtype).split(".")[-1], "float16")
            state = optimizer.state_dict()["state"][0]
            self.assertEqual(state["exp_avg"].dtype, np.float32)
            self.assertEqual(state["exp_avg_sq"].dtype, np.float32)
            self.assertEqual(state["master_weight"].dtype, np.float32)
            self.assertTrue(np.isfinite(parameter.numpy()).all())

    def test_scaler_matches_unscaled_and_detects_overflow(self):
        for device in ts.available_devices():
            a = nn.Parameter(ts.tensor([1.0, 2.0], device=device))
            b = nn.Parameter(a.numpy().copy(), device=device)
            optimizer = ts.optim.SGD([a], lr=0.1)
            reference = ts.optim.SGD([b], lr=0.1)
            scaler = ts.amp.GradScaler(init_scale=16, growth_interval=2)
            for _ in range(2):
                optimizer.zero_grad()
                reference.zero_grad()
                scaler.scale((a**2).sum()).backward()
                (b**2).sum().backward()
                scaler.unscale_(optimizer)
                with self.assertRaises(RuntimeError):
                    scaler.unscale_(optimizer)
                self.assertTrue(scaler.step(optimizer))
                reference.step()
                scaler.update()
            self.assertEqual(scaler.get_scale(), 32)
            np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-6)
            before = a.numpy().copy()
            a.grad = ts.tensor([np.inf, 1.0], device=device)
            self.assertFalse(scaler.step(optimizer))
            scaler.update()
            self.assertEqual(scaler.get_scale(), 16)
            np.testing.assert_array_equal(a.numpy(), before)
            restored = ts.amp.GradScaler()
            restored.load_state_dict(scaler.state_dict())
            self.assertEqual(restored.state_dict(), scaler.state_dict())

    def test_fp16_language_model_train_step(self):
        for device in ts.available_devices():
            model = nn.TransformerLM(
                nn.TransformerConfig(8, dim=16, num_heads=2, num_layers=1, max_seq_len=8),
                device=device,
            ).to(dtype="float16")
            optimizer = ts.optim.AdamW(model.parameters(), lr=0.01)
            scaler = ts.amp.GradScaler(init_scale=8)
            x = ts.tensor([[1, 2, 3, 4]], device=device)
            y = ts.tensor([[2, 3, 4, 5]], device=device)
            before = model.loss(x, y).item()
            for _ in range(3):
                optimizer.zero_grad()
                loss = model.loss(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.clip_grad_norm_(model.parameters(), 1)
                self.assertTrue(scaler.step(optimizer))
                scaler.update()
                ts.evaluate(loss, list(model.parameters()), optimizer.state)
            self.assertLess(model.loss(x, y).item(), before)

    def test_warmup_cosine_and_scheduler_resume(self):
        parameter = nn.Parameter([1.0])
        optimizer = ts.optim.SGD([parameter], lr=1)
        scheduler = ts.optim.WarmupCosineLR(optimizer, warmup_steps=2, total_steps=6, min_lr=0.1)
        rates = [scheduler.get_last_lr()[0]]
        for _ in range(8):
            scheduler.step()
            rates.append(scheduler.get_last_lr()[0])
        np.testing.assert_allclose(rates[:3], [0.5, 1, 1])
        self.assertEqual(rates[-1], 0.1)
        restored = ts.optim.WarmupCosineLR(optimizer, warmup_steps=0, total_steps=20)
        restored.load_state_dict(scheduler.state_dict())
        self.assertEqual(restored.state_dict(), scheduler.state_dict())
        with self.assertRaises(ValueError):
            ts.optim.WarmupCosineLR(optimizer, warmup_steps=4, total_steps=4)

    def test_gradient_accumulation_matches_full_batch(self):
        model = nn.Linear(3, 4)
        x = ts.randn(6, 3)
        targets = ts.tensor([0, 1, 2, 3, 0, 1])
        F = nn.functional
        F.cross_entropy(model(x), targets).backward()
        expected = [p.grad.numpy().copy() for p in model.parameters()]
        model.zero_grad()
        for i in range(3):
            (
                F.cross_entropy(model(x[2 * i : 2 * i + 2]), targets[2 * i : 2 * i + 2]) / 3
            ).backward()
        for actual, wanted in zip(model.parameters(), expected):
            np.testing.assert_allclose(actual.grad.numpy(), wanted, atol=1e-7)

    def test_nested_training_checkpoint_and_legacy_loading(self):
        model = nn.Linear(3, 4)
        optimizer = ts.optim.AdamW(model.parameters())
        scheduler = ts.optim.WarmupCosineLR(optimizer, 2, 10)
        model(ts.randn(2, 3)).sum().backward()
        optimizer.step()
        scheduler.step()
        scaler = ts.amp.GradScaler()
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.npz"
            ts.save(state, path)
            loaded = ts.load(path)
            self.assertEqual(loaded["step"], 1)
            self.assertIsInstance(loaded["optimizer"]["param_groups"][0]["betas"], tuple)
            self.assertIn(0, loaded["optimizer"]["state"])
            clone = nn.Linear(3, 4)
            clone.load_state_dict(loaded["model"])
            restored = ts.optim.AdamW(clone.parameters())
            restored.load_state_dict(loaded["optimizer"])
            np.testing.assert_array_equal(model.weight.numpy(), clone.weight.numpy())
            legacy = Path(directory) / "legacy.npz"
            np.savez(legacy, **model.state_dict())
            self.assertEqual(set(ts.load(legacy)), set(model.state_dict()))
            with self.assertRaises(TypeError):
                ts.save({"unsafe": np.array([object()], dtype=object)}, path)
            self.assertEqual(ts.load(path)["step"], 1)

    def test_model_loading_preserves_destination_precision(self):
        source = nn.Linear(3, 4)
        for device in ts.available_devices():
            destination = nn.Linear(3, 4).to(device, dtype="float16")
            destination.load_state_dict(source.state_dict())
            self.assertIn("float16", str(destination.weight.dtype))
            np.testing.assert_allclose(destination.weight.numpy(), source.weight.numpy(), atol=3e-4)


if __name__ == "__main__":
    unittest.main()
