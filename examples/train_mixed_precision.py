"""Train a small classifier using operator autocast and FP32 parameters."""

import argparse

import numpy as np

import tensorsmith as ts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    ts.seed(42)
    rng = np.random.default_rng(8)
    inputs = rng.normal(size=(64, 8)).astype(np.float32)
    labels = (inputs @ rng.normal(size=(8, 4))).argmax(1)
    model = ts.nn.Sequential(ts.nn.Linear(8, 32), ts.nn.GELU(), ts.nn.Linear(32, 4)).to(args.device)
    optimizer = ts.optim.AdamW(model.parameters(), lr=0.01)
    scaler = ts.amp.GradScaler(init_scale=128)
    x, y = ts.tensor(inputs, device=args.device), ts.tensor(labels, device=args.device)
    for step in range(args.steps):
        optimizer.zero_grad()
        with ts.amp.autocast(args.device):
            logits = model(x)
            loss = ts.nn.CrossEntropyLoss()(logits, y)
        scaler.scale(loss).backward()
        updated = scaler.step(optimizer)
        scaler.update()
        ts.evaluate(loss, list(model.parameters()), optimizer.state)
        if step % 5 == 0 or step == args.steps - 1:
            print(
                f"step={step + 1} loss={loss.item():.4f} logits={logits.dtype} parameters={model.layers[0].weight.dtype} updated={updated}"
            )


if __name__ == "__main__":
    main()
