"""Train-only empirical-prior baselines, using the task evaluation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from task_data import DEFAULT_CACHE, language_batches, load_cifar, load_wikitext


def run(args):
    language, vocabulary, text_provenance = load_wikitext(args.cache)
    images, image_provenance = load_cifar(args.cache, args.seed)
    word_counts = np.bincount(language["train"], minlength=len(vocabulary)).astype(np.float64) + 1
    word_probability = word_counts / word_counts.sum()
    class_counts = np.bincount(images["train"][1], minlength=10).astype(np.float64)
    class_probability = class_counts / class_counts.sum()
    report = {
        "methodology": "Fit solely on training labels/tokens; add-one word smoothing; same fixed-stream usable targets as full task reports; no model or GPU used",
        "seed": args.seed,
        "language_batch_size": args.batch_size,
        "language_seq_len": args.seq_len,
        "provenance": {"wikitext2": text_provenance, "cifar10": image_provenance},
        "splits": {},
    }
    for split in ("valid", "test"):
        labels = np.concatenate(
            [
                y.reshape(-1)
                for _, y in language_batches(language[split], args.batch_size, args.seq_len)
            ]
        )
        nll = float(-np.log(word_probability[labels]).mean())
        image_labels = images[split][1]
        report["splits"][split] = {
            "wikitext2_train_fitted_unigram": {
                "evaluated_targets": int(labels.size),
                "nll": nll,
                "perplexity": float(np.exp(nll)),
            },
            "cifar10_train_fitted_class_prior": {
                "evaluated_targets": int(image_labels.size),
                "nll": float(-np.log(class_probability[image_labels]).mean()),
                "accuracy": float(np.mean(image_labels == np.argmax(class_probability))),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/train-fitted-baselines.json")
    )
    options = parser.parse_args()
    if min(options.batch_size, options.seq_len) <= 0:
        raise ValueError("batch size and sequence length must be positive")
    print(json.dumps(run(options)["splits"], indent=2))
