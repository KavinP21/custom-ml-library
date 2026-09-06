import gc
import unittest
import weakref

import numpy as np

import tensorsmith as ts


class HookTests(unittest.TestCase):
    def test_checkpoint_does_not_apply_parameter_hook_twice(self):
        model = ts.nn.Linear(2, 1, bias=False)
        model.weight._data[...] = 1
        calls = []

        def hook(gradient):
            calls.append(gradient.numpy().copy())
            return gradient * 2

        model.weight.register_hook(hook)
        ts.nn.checkpoint(model, ts.ones(1, 2)).sum().backward()
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(model.weight.grad.numpy(), [[2, 2]])

    def test_gradient_hook_sees_summed_branches_and_changes_vjp(self):
        x = ts.tensor([2.0, 3.0], requires_grad=True)
        y = x * x
        seen = []
        y.register_hook(lambda gradient: seen.append(gradient.numpy().copy()))
        with y.register_hook(lambda gradient: gradient * 2):
            (y + 3 * y).sum().backward(retain_graph=True)
        np.testing.assert_array_equal(seen[0], [4, 4])
        np.testing.assert_array_equal(x.grad.numpy(), [32, 48])
        x.zero_grad()
        y.sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(), [4, 6])

    def test_grad_function_runs_hooks_without_accumulating(self):
        x = ts.tensor([2.0], requires_grad=True)
        x.register_hook(lambda gradient: gradient * 3)
        (gradient,) = ts.grad(x * x, x)
        np.testing.assert_array_equal(gradient.numpy(), [12])
        self.assertIsNone(x.grad)

    def test_hook_contract_and_weak_handle(self):
        x = ts.ones(2, requires_grad=True)
        handle = x.register_hook(lambda gradient: ts.ones(3))
        with self.assertRaisesRegex(ValueError, "shape/dtype/device"):
            x.sum().backward()
        handle.remove()
        handle.remove()
        reference = weakref.ref(x)
        del x
        gc.collect()
        self.assertIsNone(reference())
        handle.remove()

    def test_module_hooks_transform_inputs_and_outputs(self):
        model = ts.nn.Linear(2, 1, bias=False)
        model.weight._data[...] = 1
        events = []
        pre = model.register_forward_pre_hook(lambda module, args: (args[0] * 2,))
        post = model.register_forward_hook(
            lambda module, args, output: events.append(output.item())
        )
        with model.register_forward_hook(lambda module, args, output: output + 1):
            result = model(ts.ones(1, 2))
        self.assertEqual(result.item(), 5)
        self.assertEqual(events, [4])
        pre.remove()
        post.remove()
        self.assertEqual(model(ts.ones(1, 2)).item(), 2)

    def test_keyword_hooks_and_registry_not_in_state(self):
        class Scale(ts.nn.Module):
            def forward(self, x, *, factor=1):
                return x * factor

        model = Scale()
        model.register_forward_pre_hook(
            lambda module, args, kwargs: (args, {"factor": 3}), with_kwargs=True
        )
        self.assertEqual(model(ts.ones(1), factor=2).item(), 3)
        self.assertEqual(list(model.named_parameters()), [])
        self.assertEqual(list(model.named_modules()), [("", model)])
        self.assertEqual(model.state_dict(), {})
