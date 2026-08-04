from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _load_experiments(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("manifest must contain a non-empty experiments list")
    for experiment in experiments:
        if not isinstance(experiment, dict) or not {"name", "model_config"} <= experiment.keys():
            raise ValueError("each experiment requires name and model_config")
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible baseline and ablation studies")
    parser.add_argument("--manifest", default="configs/experiments-paper.json")
    parser.add_argument("--train-config", default="data/train-real.json")
    parser.add_argument("--output-root", default="outputs/paper")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--stage", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--compile-mode",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        default="none",
    )
    parser.add_argument("--qlib-provider-uri", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--skip-qlib", action="store_true")
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Resume training even when a completed train_summary.json exists",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    positive_arguments = {
        "batch-size": args.batch_size,
        "train-stride": args.train_stride,
        "epochs": args.epochs,
        "patience": args.patience,
    }
    for name, value in positive_arguments.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must not contain duplicates")

    experiments = _load_experiments(args.manifest)
    if args.experiments:
        selected = set(args.experiments)
        experiments = [item for item in experiments if item["name"] in selected]
        missing = selected - {item["name"] for item in experiments}
        if missing:
            raise ValueError(f"unknown experiments: {', '.join(sorted(missing))}")

    run_directories: list[Path] = []
    for experiment in experiments:
        for seed in args.seeds:
            run_dir = Path(args.output_root) / str(experiment["name"]) / f"seed-{seed}"
            run_directories.append(run_dir)
            if args.stage in {"all", "train"}:
                completed = (
                    (run_dir / "train_summary.json").exists()
                    and (run_dir / "best.safetensors").exists()
                )
                if completed and not args.force_train:
                    print(f"skip completed training: {run_dir}", flush=True)
                else:
                    train_command = [
                        sys.executable,
                        "scripts/train.py",
                        "--model-config",
                        str(experiment["model_config"]),
                        "--train-config",
                        args.train_config,
                        "--output-dir",
                        str(run_dir),
                        "--batch-size",
                        str(args.batch_size),
                        "--num-workers",
                        str(args.num_workers),
                        "--train-stride",
                        str(args.train_stride),
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(args.epochs),
                        "--patience",
                        str(args.patience),
                        "--compile-mode",
                        args.compile_mode,
                    ]
                    if args.device is not None:
                        train_command.extend(["--device", args.device])
                    checkpoint = run_dir / "last.safetensors"
                    if checkpoint.exists():
                        train_command.extend(["--resume", str(run_dir)])
                    _run(train_command, args.dry_run)

            if args.stage in {"all", "evaluate"}:
                evaluate_command = [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--checkpoint-dir",
                    str(run_dir),
                    "--split",
                    "test",
                    "--qlib-provider-uri",
                    args.qlib_provider_uri,
                ]
                if args.device is not None:
                    evaluate_command.extend(["--device", args.device])
                if args.skip_qlib:
                    evaluate_command.append("--skip-qlib")
                _run(evaluate_command, args.dry_run)

    if args.stage in {"all", "evaluate"}:
        compare_command = [
            sys.executable,
            "scripts/compare_experiments.py",
            "--output-root",
            args.output_root,
            "--run-dirs",
            *(str(path) for path in run_directories),
        ]
        _run(compare_command, args.dry_run)


if __name__ == "__main__":
    main()
