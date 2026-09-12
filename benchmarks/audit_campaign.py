"""Read-only audit of completed real-task reports and comparison consistency."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text())


def audit(results):
    campaign = read(results / "campaign-status.json")
    require(campaign["status"] == "complete", "campaign is not complete")
    require(
        all(job["status"] == "complete" and job["exit_code"] == 0 for job in campaign["jobs"]),
        "campaign contains an incomplete/failed job",
    )
    require(
        [job["name"] for job in campaign["jobs"]] == campaign["order"],
        "campaign job coverage/order differs from plan",
    )
    summary = {"quality": {}, "acceptance": {}, "isolated_speed": {}, "notes": []}
    non_model_arguments = {"engine", "output", "checkpoint", "cache", "log_interval"}
    for task, epochs, train_targets, test_targets in (
        ("cifar10", 30, 1_350_000, 10_000),
        ("wikitext2", 10, 20_886_080, 245_552),
    ):
        reports = {
            engine: read(results / f"{task}-{engine}-metal-seed42.json")
            for engine in ("tensorsmith", "torch")
        }
        actual, reference = reports.values()
        require(actual["dataset"] == reference["dataset"], f"{task}: data mismatch")
        require(actual["model"] == reference["model"], f"{task}: architecture mismatch")
        for key, value in actual["arguments"].items():
            if key not in non_model_arguments:
                require(value == reference["arguments"][key], f"{task}: mismatched {key}")
        if actual.get("initial_state_sha256"):
            require(
                actual["initial_state_sha256"] == reference["initial_state_sha256"],
                f"{task}: different initial weights",
            )
        else:
            summary["notes"].append(
                f"{task}: historical TensorSmith run predates initial-weight hash recording; both use the same seeded CPU initializer, but a historical hash is unavailable"
            )
        summary["quality"][task] = {}
        for engine, report in reports.items():
            require(report["status"] == "complete", f"{task}/{engine}: incomplete")
            require(report["arguments"]["max_steps"] == 0, "capped quality run")
            require(len(report["epochs"]) == epochs, "missing full epochs")
            require(report["total_training_targets"] == train_targets, "missing training targets")
            require(report["test"]["evaluated_targets"] == test_targets, "missing test targets")
            selected = min(report["epochs"], key=lambda epoch: epoch["validation"]["nll"])
            require(selected["epoch"] == report["best_epoch"], "checkpoint selection mismatch")
            require(math.isfinite(report["test"]["nll"]), "non-finite test NLL")
            summary["quality"][task][engine] = {
                "test": report["test"],
                "best_epoch": report["best_epoch"],
                "training_and_evaluation_seconds": report["training_and_evaluation_seconds"],
            }
        summary["isolated_speed"][task] = {}
        for engine in reports:
            repeats = [
                read(results / f"{task}-isolated-{engine}-metal-r{repeat}.json")
                for repeat in range(1, 4)
            ]
            phases = {}
            for repeat in repeats:
                require(repeat["engine"] == engine, "timing engine mismatch")
                require(
                    Path(repeat["quality_checkpoint"]).name
                    == f"{task}-tensorsmith-metal-seed42.npz",
                    "timing does not use the same trained weights",
                )
                for phase in repeat["results"]:
                    require(
                        phase["iterations"] == 50 and phase["warmup"] == 10,
                        "timing budget mismatch",
                    )
                    require(len(phase["samples_ms"]) == 50, "missing timing samples")
                    require(
                        all(math.isfinite(v) and v > 0 for v in phase["samples_ms"]),
                        "invalid timing",
                    )
                    phases.setdefault(phase["name"], []).append(phase)
            for name, values in phases.items():
                require(len(values) == 3, "missing independent repeat")
                medians = [value["median_ms"] for value in values]
                phases[name] = {
                    "median_of_run_medians_ms": statistics.median(medians),
                    "run_medians_ms": medians,
                    "run_p95_ms": [value["p95_ms"] for value in values],
                    "run_memory_after_phase": [value.get("memory_after_phase") for value in values],
                }
            summary["isolated_speed"][task][engine] = phases
    for backend, minimum in (("cpu", 16), ("metal", 20)):
        report = read(results / f"real-task-acceptance-{backend}.json")
        require(report["failed"] == 0, f"{backend}: acceptance failure")
        require(report["passed"] >= minimum, f"{backend}: missing acceptance coverage")
        require(
            all(check["status"] == "pass" for check in report["checks"]), "inconsistent pass count"
        )
        summary["acceptance"][backend] = {
            "passed": report["passed"],
            "checks": [check["name"] for check in report["checks"]],
        }
    large = read(results / "wikitext2-129m-STRESS-metal.json")
    require(large["status"] == "complete", "large execution incomplete")
    require(large["model"]["parameter_count"] >= 129_000_000, "large model missing")
    require(large["arguments"]["max_steps"] == 5, "stress budget changed")
    summary["large_stress"] = {
        "parameters": large["model"]["parameter_count"],
        "training_targets": large["total_training_targets"],
        "note": "Five real-text updates plus independent full-gradient acceptance; not a convergence/quality result",
    }
    optimized_paths = [
        results / f"wikitext2-optimized-pair-{engine}-metal-r{repeat}.json"
        for engine in ("tensorsmith", "torch")
        for repeat in range(1, 4)
    ]
    if any(path.exists() for path in optimized_paths):
        require(
            all(path.exists() for path in optimized_paths),
            "incomplete fused-baseline paired repeats",
        )
        summary["optimized_language_pair"] = {}
        for engine in ("tensorsmith", "torch"):
            repeats = [
                read(results / f"wikitext2-optimized-pair-{engine}-metal-r{repeat}.json")
                for repeat in range(1, 4)
            ]
            for repeat in repeats:
                require(repeat["engine"] == engine, "optimized pair engine mismatch")
                require(repeat["dataset_input_shape"] == [16, 128], "optimized pair shape mismatch")
                require(
                    repeat["arguments"]["torch_fused_optimizer"] == (engine == "torch"),
                    "optimized optimizer variant not recorded correctly",
                )
                require(
                    Path(repeat["quality_checkpoint"]).name
                    == "wikitext2-tensorsmith-metal-seed42.npz",
                    "optimized pair weights mismatch",
                )
                for phase in repeat["results"]:
                    require(
                        phase["iterations"] == 50 and phase["warmup"] == 10,
                        "optimized budget mismatch",
                    )
                    require(len(phase["samples_ms"]) == 50, "optimized samples missing")
                    require(
                        all(math.isfinite(v) and v > 0 for v in phase["samples_ms"]),
                        "invalid optimized timing",
                    )
            summary["optimized_language_pair"][engine] = {
                name: {
                    "run_medians_ms": [
                        next(p["median_ms"] for p in r["results"] if p["name"] == name)
                        for r in repeats
                    ],
                    "median_of_run_medians_ms": statistics.median(
                        next(p["median_ms"] for p in r["results"] if p["name"] == name)
                        for r in repeats
                    ),
                    "run_p95_ms": [
                        next(p["p95_ms"] for p in r["results"] if p["name"] == name)
                        for r in repeats
                    ],
                    "run_memory_after_phase": [
                        next(p.get("memory_after_phase") for p in r["results"] if p["name"] == name)
                        for r in repeats
                    ],
                }
                for name in (
                    "real_data_complete_training_update",
                    "real_data_full_logits_inference",
                )
            }
    summary["audit"] = "pass"
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--output", type=Path, help="Optionally save the audited summary as JSON")
    options = parser.parse_args()
    encoded = json.dumps(audit(options.results), indent=2)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded + "\n")
    print(encoded)
