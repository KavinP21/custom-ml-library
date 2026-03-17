"""Train a tiny CNN on a synthetic, learnable image task without external datasets."""

import numpy as np

import tensorsmith as ts
from tensorsmith import nn


def make_dataset(size=256):
    images = np.random.normal(0, 0.15, (size, 1, 8, 8)).astype(np.float32)
    labels = np.random.randint(0, 2, size=size)
    # Class zero has a vertical stroke; class one has a horizontal stroke.
    images[labels == 0, 0, :, 3:5] += 1
    images[labels == 1, 0, 3:5, :] += 1
    return ts.tensor(images), ts.tensor(labels)


def main():
    ts.seed(7)
    compute = "metal" if ts.is_available("metal") else "cuda" if ts.is_available("cuda") else "cpu"
    images, labels = make_dataset()
    loader = ts.data.DataLoader(
        ts.data.TensorDataset(images, labels), batch_size=32, shuffle=True, seed=7
    )
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 2),
    ).to(compute)
    optimizer = ts.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(8):
        total = 0.0
        for x, y in loader:
            x, y = x.to(compute), y.to(compute)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch={epoch + 1} mean_loss={total / len(loader):.5f}")

    predictions = model(images.to(compute)).argmax(1).cpu().numpy()
    print(f"device={compute} accuracy={(predictions == labels.numpy()).mean():.1%}")


if __name__ == "__main__":
    main()
