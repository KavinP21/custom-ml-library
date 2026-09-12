"""Real public datasets, verified downloads, and deterministic task protocols.

No pickle deserialization or archive extraction is required. Dataset files and
large checkpoints are deliberately excluded from the source distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import ssl
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
DEFAULT_CACHE = Path("benchmarks/data")
CIFAR_HF_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"
CIFAR_HF_HASHES = {
    "train": "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",
    "test": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
}


def open_url(request, timeout):
    # Some python.org macOS installs omit their bundled roots. Use the
    # system's trusted CA bundle in that case, never unverified TLS.
    context = ssl.create_default_context()
    if (
        ssl.get_default_verify_paths().cafile is None
        and not os.environ.get("SSL_CERT_FILE")
        and Path("/etc/ssl/cert.pem").exists()
    ):
        context.load_verify_locations("/etc/ssl/cert.pem")
    return urllib.request.urlopen(request, timeout=timeout, context=context)


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "TensorSmith-task-benchmark"})
    print(f"Downloading {url}", flush=True)
    with open_url(request, timeout=120) as response, temporary.open("wb") as out:
        while block := response.read(1024 * 1024):
            out.write(block)
    temporary.replace(path)


def prepare(cache=DEFAULT_CACHE, task="all", revision="main", cifar_source="huggingface"):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if task in {"all", "cifar10"} and cifar_source == "huggingface":
        files = {}
        for split, checksum in CIFAR_HF_HASHES.items():
            path = cache / f"cifar10-{split}.parquet"
            url = f"https://huggingface.co/datasets/uoft-cs/cifar10/resolve/{CIFAR_HF_REVISION}/plain_text/{split}-00000-of-00001.parquet"
            if not path.exists():
                download(url, path)
            if digest(path) != checksum:
                raise ValueError("CIFAR Parquet file failed its published SHA256")
            files[split] = {"url": url, "sha256": checksum}
        manifest["cifar10"] = {"storage": "parquet", "revision": CIFAR_HF_REVISION, "files": files}
    elif task in {"all", "cifar10"}:
        path = cache / "cifar-10-binary.tar.gz"
        if not path.exists():
            download(CIFAR_URL, path)
        if digest(path, "md5") != CIFAR_MD5:
            raise ValueError("CIFAR archive does not match the publisher's checksum")
        manifest["cifar10"] = {"url": CIFAR_URL, "md5": CIFAR_MD5, "sha256": digest(path)}
    if task in {"all", "wikitext2"}:
        existing = manifest.get("wikitext2")
        if existing is None:
            request = urllib.request.Request(
                f"https://api.github.com/repos/pytorch/examples/commits/{revision}",
                headers={"User-Agent": "TensorSmith-task-benchmark"},
            )
            with open_url(request, timeout=60) as response:
                commit = json.load(response)["sha"]
            existing = {"revision": commit, "files": {}}
        for split in ("train", "valid", "test"):
            path = cache / "wikitext2" / f"{split}.txt"
            url = (
                "https://raw.githubusercontent.com/pytorch/examples/"
                f"{existing['revision']}/word_language_model/data/wikitext-2/{split}.txt"
            )
            if not path.exists():
                download(url, path)
            checksum = digest(path)
            recorded = existing["files"].get(split)
            if recorded and checksum != recorded["sha256"]:
                raise ValueError(f"WikiText {split} failed the recorded checksum")
            existing["files"][split] = {"url": url, "sha256": checksum}
        manifest["wikitext2"] = existing
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def provenance(cache, task):
    path = Path(cache) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("Run python benchmarks/task_data.py --task all first")
    return json.loads(path.read_text())[task]


def load_cifar(cache=DEFAULT_CACHE, seed=42):
    cache = Path(cache)
    recorded = provenance(cache, "cifar10")
    if recorded.get("storage") == "parquet":
        import pyarrow.parquet as pq
        from PIL import Image

        decoded_path = cache / "cifar10-decoded.npz"
        decoded_meta = cache / "cifar10-decoded.json"
        source_hashes = [recorded["files"][split]["sha256"] for split in ("train", "test")]
        for split in ("train", "test"):
            if digest(cache / f"cifar10-{split}.parquet") != CIFAR_HF_HASHES[split]:
                raise ValueError("CIFAR Parquet checksum mismatch")
        if decoded_path.exists() and decoded_meta.exists():
            metadata = json.loads(decoded_meta.read_text())
            if (
                metadata["source_sha256"] == source_hashes
                and digest(decoded_path) == metadata["sha256"]
            ):
                with np.load(decoded_path, allow_pickle=False) as arrays:
                    return split_cifar(
                        arrays["train_images"],
                        arrays["train_labels"],
                        arrays["test_images"],
                        arrays["test_labels"],
                        recorded,
                        seed,
                    )
        images, labels = [], []
        for split in ("train", "test"):
            path = cache / f"cifar10-{split}.parquet"
            if digest(path) != CIFAR_HF_HASHES[split]:
                raise ValueError("CIFAR Parquet checksum mismatch")
            data = pq.read_table(path).to_pydict()
            expected_count = 50000 if split == "train" else 10000
            if len(data["label"]) != expected_count:
                raise ValueError("wrong CIFAR split size")
            pixels = []
            for record in data["img"]:
                with Image.open(io.BytesIO(record["bytes"])) as image:
                    pixels.append(np.asarray(image.convert("RGB")).transpose(2, 0, 1))
            images.append(np.stack(pixels))
            labels.append(np.asarray(data["label"], np.int64))
        np.savez(
            decoded_path,
            train_images=images[0],
            train_labels=labels[0],
            test_images=images[1],
            test_labels=labels[1],
        )
        decoded_meta.write_text(
            json.dumps({"source_sha256": source_hashes, "sha256": digest(decoded_path)})
        )
        return split_cifar(images[0], labels[0], images[1], labels[1], recorded, seed)
    archive = cache / "cifar-10-binary.tar.gz"
    if digest(archive) != recorded["sha256"] or digest(archive, "md5") != CIFAR_MD5:
        raise ValueError("CIFAR checksum mismatch")
    images, labels = [], []
    with tarfile.open(archive, "r:gz") as stream:
        for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
            member = stream.getmember(f"cifar-10-batches-bin/{name}.bin")
            if not member.isfile() or member.size != 10000 * 3073:
                raise ValueError("invalid CIFAR binary member")
            with stream.extractfile(member) as data:
                records = np.frombuffer(data.read(), dtype=np.uint8).reshape(10000, 3073)
            labels.append(records[:, 0].astype(np.int64))
            images.append(records[:, 1:].reshape(-1, 3, 32, 32).copy())
    train_x, train_y = np.concatenate(images[:5]), np.concatenate(labels[:5])
    return split_cifar(train_x, train_y, images[-1], labels[-1], recorded, seed)


def split_cifar(train_x, train_y, test_x, test_y, recorded, seed):
    if train_x.shape != (50000, 3, 32, 32) or test_x.shape != (10000, 3, 32, 32):
        raise ValueError("invalid full CIFAR arrays")
    if not np.array_equal(
        np.bincount(train_y, minlength=10), np.full(10, 5000)
    ) or not np.array_equal(np.bincount(test_y, minlength=10), np.full(10, 1000)):
        raise ValueError("invalid CIFAR label counts")
    # A fixed stratified validation split: 500 images/class, never from test.
    rng = np.random.default_rng(seed)
    valid_ids = np.concatenate(
        [rng.permutation(np.flatnonzero(train_y == c))[:500] for c in range(10)]
    )
    train_ids = np.setdiff1d(np.arange(len(train_y)), valid_ids)
    splits = {
        "train": (train_x[train_ids], train_y[train_ids]),
        "valid": (train_x[valid_ids], train_y[valid_ids]),
        "test": (test_x, test_y),
    }
    recorded = dict(
        recorded,
        validation_index_sha256=hashlib.sha256(valid_ids.tobytes()).hexdigest(),
        decoded_pixel_label_sha256=hashlib.sha256(
            train_x.tobytes() + train_y.tobytes() + test_x.tobytes() + test_y.tobytes()
        ).hexdigest(),
    )
    return splits, recorded


def image_batch(images, rng=None):
    """Identical CPU augmentation/input transfer for both measured frameworks."""
    x = images.astype(np.float32) / 255
    if rng is not None:
        padded = np.pad(x, ((0, 0), (0, 0), (4, 4), (4, 4)), mode="constant")
        crops = rng.integers(0, 9, (len(x), 2))
        x = np.stack([padded[i, :, y : y + 32, z : z + 32] for i, (y, z) in enumerate(crops)])
        flip = rng.random(len(x)) < 0.5
        x[flip] = x[flip, :, :, ::-1]
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)
    return np.ascontiguousarray((x - mean) / std)


def encode_words(text, vocabulary):
    return np.asarray(
        [
            vocabulary.get(word, vocabulary["<unk>"])
            for line in text.splitlines()
            for word in line.split() + ["<eos>"]
        ],
        dtype=np.int64,
    )


def load_wikitext(cache=DEFAULT_CACHE):
    cache = Path(cache)
    recorded = provenance(cache, "wikitext2")
    texts = {}
    for split in ("train", "valid", "test"):
        path = cache / "wikitext2" / f"{split}.txt"
        if digest(path) != recorded["files"][split]["sha256"]:
            raise ValueError(f"WikiText {split} checksum mismatch")
        texts[split] = path.read_text(encoding="utf8")
    # Learn vocabulary ONLY from training data, with explicit OOV mapping.
    vocabulary = {"<unk>": 0, "<eos>": 1}
    for word in texts["train"].split():
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
    tokens = {split: encode_words(text, vocabulary) for split, text in texts.items()}
    return tokens, list(vocabulary), recorded


def language_batches(tokens, batch_size, seq_len):
    """Fixed contiguous streams, resetting attention context every segment.

    All usable shifted targets are visited once, including the short final
    segment. At most batch_size-1 trailing corpus tokens are discarded.
    """
    usable = len(tokens) // batch_size * batch_size
    streams = tokens[:usable].reshape(batch_size, -1)
    for start in range(0, streams.shape[1] - 1, seq_len):
        end = min(start + seq_len, streams.shape[1] - 1)
        yield streams[:, start:end].copy(), streams[:, start + 1 : end + 1].copy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download checksum-verified CIFAR-10 and WikiText-2"
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--task", choices=["all", "cifar10", "wikitext2"], default="all")
    parser.add_argument(
        "--revision", default="main", help="PyTorch examples revision; resolved to immutable SHA"
    )
    parser.add_argument("--cifar-source", choices=["binary", "huggingface"], default="huggingface")
    options = parser.parse_args()
    prepare(options.cache, options.task, options.revision, options.cifar_source)
