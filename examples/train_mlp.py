"""Train an XOR classifier with the complete TensorSmith stack."""

import tensorsmith as ts
from tensorsmith import nn


def main():
    ts.seed(42)
    compute = "metal" if ts.is_available("metal") else "cuda" if ts.is_available("cuda") else "cpu"
    x = ts.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=compute)
    y = ts.tensor([0, 1, 1, 0], device=compute)
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2)).to(compute)
    optimizer = ts.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)

    for step in range(301):
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step={step:03d} loss={loss.item():.5f}")

    predictions = model(x).argmax(1).cpu().tolist()
    print(f"device={compute} predictions={predictions}")


if __name__ == "__main__":
    main()
