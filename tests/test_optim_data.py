import unittest


import numpy as np


import tensorsmith as ts


from tensorsmith import nn


class OptimizerAndDataTests(unittest.TestCase):
    def test_sgd_learns_linear_regression(self):
        np.random.seed(7)
        x = ts.tensor(np.random.randn(64, 2).astype(np.float32))
        target = x @ ts.tensor([[2.0], [-3.0]]) + 0.5
        model = nn.Linear(2, 1)
        optimizer = ts.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)
        initial = nn.MSELoss()(model(x), target).item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(x), target)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * 1e-5)

    def test_data_loader(self):
        dataset = ts.data.TensorDataset(ts.arange(10), ts.arange(10) * 2)
        loader = ts.data.DataLoader(dataset, batch_size=4)
        batches = list(loader)
        self.assertEqual([batch[0].shape for batch in batches], [(4,), (4,), (2,)])
        np.testing.assert_array_equal(batches[-1][1].numpy(), [16, 18])
