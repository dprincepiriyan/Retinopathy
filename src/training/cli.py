"""Command-line entry point for training and calibration.

Examples::

    # Stage-1 triage (reads configs/tier2_stage1_triage.yaml)
    python -m src.training.cli stage1 --data-root data/processed --out checkpoints

    # Stage-2 grading, 4-fold CV (reads configs/tier3_stage2_grading.yaml)
    python -m src.training.cli stage2 --data-root data/processed --out checkpoints --folds 1 2

    # Calibrate thresholds / temperature / conformal from OOF + test predictions
    python -m src.training.cli calibrate --out checkpoints

    # Override any config value with dotted keys
    python -m src.training.cli stage2 --set model.backbone_name=convnext_tiny training.ema_decay=null
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.utils.config import load_yaml_config

STAGE1_CONFIG = "configs/tier2_stage1_triage.yaml"
STAGE2_CONFIG = "configs/tier3_stage2_grading.yaml"


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply `a.b.c=value` overrides (values parsed as YAML: numbers, null, lists...)."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RetinaNet-Pro training CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, default_cfg in (("stage1", STAGE1_CONFIG), ("stage2", STAGE2_CONFIG)):
        p = sub.add_parser(name)
        p.add_argument("--config", default=default_cfg)
        p.add_argument("--data-root", default="data/processed")
        p.add_argument("--out", default="checkpoints")
        p.add_argument("--mirror", default=None, help="Extra directory to mirror checkpoints into (e.g. Drive)")
        p.add_argument("--set", nargs="*", default=[], help="Dotted config overrides, e.g. training.epochs=3")
        p.add_argument("--no-resume", action="store_true")
        if name == "stage2":
            p.add_argument("--folds", nargs="*", type=int, default=None)

    c = sub.add_parser("calibrate")
    c.add_argument("--out", default="checkpoints")
    c.add_argument("--alpha", type=float, default=0.05)

    m = sub.add_parser("build-stage1-manifest")
    m.add_argument("--data-root", default="data/processed")

    args = parser.parse_args(argv)

    if args.cmd == "stage1":
        from src.training.stage1 import train_stage1

        cfg = apply_overrides(load_yaml_config(args.config)["stage1_triage"], args.set)
        train_stage1(cfg, args.data_root, args.out, args.mirror, resume=not args.no_resume)
    elif args.cmd == "stage2":
        from src.training.stage2 import run_stage2_cv

        cfg = apply_overrides(load_yaml_config(args.config)["stage2_grading"], args.set)
        run_stage2_cv(cfg, args.data_root, args.out, args.mirror, folds=args.folds)
    elif args.cmd == "calibrate":
        from src.training.calibration import calibrate_from_files

        out = Path(args.out)
        payload = calibrate_from_files(
            out / "stage2_oof.npz", out / "calibration_params.json", out / "stage2_test_ensemble.npz", args.alpha
        )
        print(json.dumps(payload, indent=2))
    elif args.cmd == "build-stage1-manifest":
        from src.data.stage1_manifest import COMBINED_MANIFEST_NAME, build_stage1_combined_manifest

        root = Path(args.data_root)
        summary = build_stage1_combined_manifest(
            root / "stage1_triage_splits.csv", root / "stage2_grading_splits.csv", root / COMBINED_MANIFEST_NAME
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
