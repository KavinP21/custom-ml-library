"""Offline benchmark protocol tests: quality metrics must not silently cheat."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import tensorsmith as ts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from bench_tasks import learning_rate
from task_data import encode_words, image_batch, language_batches
from task_models import CifarResNet, torch_reference
from verify_tasks import resume


class TaskProtocolTests(unittest.TestCase):
    def test_language_targets_shift_with_short_tail(self):
        tokens = np.arange(43)
        batches = list(language_batches(tokens, 3, 4))
        self.assertEqual([x.shape[1] for x, _ in batches], [4, 4, 4, 1])
        self.assertEqual(sum(y.size for _, y in batches), 39)
        for x, y in batches:
            np.testing.assert_array_equal(x + 1, y)

    def test_unknown_words_and_empty_lines_keep_eos(self):
        encoded = encode_words("known missing\n\nknown\n", {"<unk>": 0, "<eos>": 1, "known": 2})
        np.testing.assert_array_equal(encoded, [2, 0, 1, 1, 2, 1])

    def test_augmentation_is_reproducible_not_applied_at_eval(self):
        raw = np.arange(2 * 3 * 32 * 32).reshape(2, 3, 32, 32).astype(np.uint8)
        a = image_batch(raw, np.random.default_rng(5))
        b = image_batch(raw, np.random.default_rng(5))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, image_batch(raw)))
        self.assertEqual(a.dtype, np.float32)

    def test_warmup_cosine_schedule(self):
        args = SimpleNamespace(warmup_steps=2, lr=0.1, min_lr=0.01)
        self.assertAlmostEqual(learning_rate(args, 0, 10), 0.05)
        self.assertAlmostEqual(learning_rate(args, 1, 10), 0.1)
        self.assertAlmostEqual(learning_rate(args, 2, 10), 0.1)
        self.assertAlmostEqual(learning_rate(args, 10, 10), 0.01)

    def test_residual_model_backward_all_parameters(self):
        model = CifarResNet(blocks_per_stage=1, width=4)
        logits = model(ts.randn(2, 3, 32, 32))
        self.assertEqual(logits.shape, (2, 10))
        logits.mean().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(np.isfinite(p.grad.numpy()).all(), name)

    def test_residual_sgd_checkpoint_continuation_and_precision(self):
        ts.seed(12)
        model = CifarResNet(blocks_per_stage=1, width=4)
        saved = {
            "model": model.state_dict(),
            "vocabulary": None,
            "config": {"seed": 12, "width": 4, "blocks_per_stage": 1},
        }
        rng = np.random.default_rng(42)
        batches = [
            (rng.normal(size=(3, 3, 32, 32)).astype(np.float32), np.array([0, 1, 2]))
            for _ in range(3)
        ]
        result = resume(saved, "cifar10", "cpu", batches)
        self.assertEqual(result["continued_updates"], 2)
        self.assertLess(max(result["continuation_loss_errors"]), 1e-6)
        logits = model(ts.randn(3, 3, 32, 32))
        self.assertEqual(logits.numpy().dtype, np.float32)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("torch"), "optional torch reference"
    )
    def test_residual_reference_running_stats_and_gradients(self):
        import torch

        torch.set_num_threads(2)
        ts.seed(42)
        model = CifarResNet(blocks_per_stage=1, width=4)
        reference = torch_reference("cifar10", model, "cpu")
        raw = np.random.randn(3, 3, 32, 32).astype(np.float32)
        actual = model(ts.tensor(raw))
        expected = reference(torch.tensor(raw))
        actual.mean().backward()
        expected.mean().backward()
        np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), atol=3e-5, rtol=2e-4)
        for name, p in model.named_parameters():
            np.testing.assert_allclose(
                p.grad.numpy(),
                reference.weights[name].grad.numpy(),
                atol=3e-5,
                rtol=2e-3,
                err_msg=name,
            )
        state = model.state_dict()
        for name, value in reference.buffers_by_name.items():
            np.testing.assert_allclose(
                state[name], value.numpy(), atol=3e-6, rtol=1e-5, err_msg=name
            )
