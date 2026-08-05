import importlib.util
import unittest

import numpy as np

import tensorsmith as ts


class ParameterGroupTests(unittest.TestCase):
    def test_group_overrides_and_unfreezing(self):
        a, b = ts.nn.Parameter([1.0]), ts.nn.Parameter([1.0])
        options = {"params": [a], "lr": 0.2}
        optimizer = ts.optim.SGD([options], lr=0.1)
        optimizer.add_param_group({"params": b, "weight_decay": 0.5})
        self.assertEqual(options, {"params": [a], "lr": 0.2})
        a.grad = ts.tensor([1.0])
        b.grad = ts.tensor([1.0])
        optimizer.step()
        np.testing.assert_allclose(a.numpy(), [0.8])
        np.testing.assert_allclose(b.numpy(), [0.85])

    def test_invalid_groups_do_not_mutate_optimizer(self):
        a, b = ts.nn.Parameter([1.0]), ts.nn.Parameter([2.0])
        optimizer = ts.optim.SGD([a])
        for group in [
            {"params": [a]},
            {"params": [b, b]},
            {"params": [b], "lr": -1},
            {"params": [b], "momentum": float("nan")},
            {"params": [b], "nesterov": True},
        ]:
            with self.assertRaises(ValueError):
                optimizer.add_param_group(group)
            self.assertEqual(len(optimizer.param_groups), 1)

    def test_group_checkpoint_continuation_and_atomic_validation(self):
        a, b = ts.nn.Parameter([1.0]), ts.nn.Parameter([2.0])
        optimizer = ts.optim.AdamW(
            [{"params": [a], "lr": 0.01}, {"params": [b], "lr": 0.1, "weight_decay": 0.2}]
        )
        for p in (a, b):
            p.grad = ts.ones(1)
        optimizer.step()
        saved = optimizer.state_dict()
        clones = [ts.nn.Parameter(p.numpy().copy()) for p in (a, b)]
        resumed = ts.optim.AdamW([{"params": [clones[0]]}, {"params": [clones[1]]}])
        resumed.load_state_dict(saved)
        for p in (a, b, *clones):
            p.grad = ts.tensor([0.3])
        optimizer.step()
        resumed.step()
        for actual, expected in zip(clones, (a, b)):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        invalid = optimizer.state_dict()
        invalid["param_groups"][1]["lr"] = -1
        with self.assertRaises(ValueError):
            resumed.load_state_dict(invalid)
        self.assertEqual(resumed.param_groups[1]["lr"], 0.1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch reference")
    def test_multigroup_updates_match_pytorch(self):
        import torch

        for actual_cls, reference_cls in [
            (ts.optim.SGD, torch.optim.SGD),
            (ts.optim.AdamW, torch.optim.AdamW),
        ]:
            values = [np.array([1.0, -2.0]), np.array([0.5, 3.0])]
            actual = [ts.nn.Parameter(v.copy(), dtype="float64") for v in values]
            reference = [torch.nn.Parameter(torch.tensor(v)) for v in values]
            groups = [{"lr": 0.02, "weight_decay": 0.1}, {"lr": 0.005, "weight_decay": 0.0}]
            kwargs = (
                {"momentum": 0.9}
                if actual_cls is ts.optim.SGD
                else {"betas": (0.8, 0.95), "amsgrad": True}
            )
            ours = actual_cls([{**g, "params": [p]} for g, p in zip(groups, actual)], **kwargs)
            theirs = reference_cls(
                [{**g, "params": [p]} for g, p in zip(groups, reference)], foreach=False, **kwargs
            )
            for step in range(5):
                for i, (a, b) in enumerate(zip(actual, reference)):
                    gradient = np.array([0.2 * (step + 1), -0.3 * (i + 1)])
                    a.grad = ts.tensor(gradient)
                    b.grad = torch.tensor(gradient)
                ours.step()
                theirs.step()
            for a, b in zip(actual, reference):
                np.testing.assert_allclose(a.numpy(), b.detach().numpy(), rtol=1e-12, atol=1e-12)
