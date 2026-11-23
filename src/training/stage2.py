"""Stage-2 ICDR grading: phased-resolution training with true K-fold out-of-fold predictions.

For every validation fold `f` in `cv.folds` (default 1-4; fold 0 is the locked test
split) a model is trained on the remaining train_val folds. Its best-epoch
predictions on fold `f` become that fold's out-of-fold (OOF) predictions, and its
predictions on the test split are stored too. After all folds:

- `stage2_oof.npz`            – OOF predictions for every train_val scan (used for
                                 threshold / temperature / conformal calibration)
- `stage2_test_ensemble.npz`  – fold-ensemble predictions on the locked test split
- `stage2_grader_fold{f}.pth` – one self-describing checkpoint per fold
                                 (the pipeline ensembles all of them)

Each fold is resumable (`stage2_fold{f}_last.pt`) and skipped once finished, so the
CV can be spread over several Colab sessions.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.data.augmentations import RetinalTrainAugment
from src.data.datasets import PreprocessedManifestDataset
from src.losses import CORALOrdinalFocalLoss, SoftQWKLoss
from src.models.factory import build_model, load_model_checkpoint, save_model_checkpoint
from src.models.pretrained import parameter_groups
from src.training.common import (
    ModelEMA,
    NonFiniteGuard,
    model_is_finite,
    balanced_sampler,
    load_training_state,
    make_loader,
    maybe_subsample,
    mirror_to,
    save_training_state,
    set_lr,
    set_seed,
    trainable_parameters,
    warmup_cosine_factor,
    write_json,
)
from src.tier4_ensemble.qwk_threshold_optimizer import NelderMeadQWKOptimizer
from src.utils.metrics import compute_clinical_metrics, quadratic_weighted_kappa
from src.utils.tpu_accelerator import ColabTPUAccelerator

DEFAULT_THRESHOLDS = (0.5, 1.5, 2.5, 3.5)


def fold_ckpt_name(fold: int) -> str:
    return f"stage2_grader_fold{fold}.pth"


def _build_coral_loss(cfg: Dict[str, Any], train_labels: np.ndarray) -> CORALOrdinalFocalLoss:
    """Apply exactly ONE imbalance correction (sampler OR loss weights), never both."""
    lcfg = cfg.get("loss", {})
    strategy = cfg.get("imbalance", {}).get("strategy", "sampler")
    gamma = float(lcfg.get("coral_focal_gamma", 1.0))
    if strategy == "loss_weights":
        # Weight each cumulative threshold by inverse prevalence of its minority side.
        pos = np.array([(train_labels > k).mean() for k in range(4)])
        minority = np.minimum(pos, 1.0 - pos).clip(1e-3, None)
        w = (1.0 / minority)
        w = w / w.mean()
        return CORALOrdinalFocalLoss(num_classes=5, gamma=gamma, alpha=0.5,
                                     threshold_weights=torch.tensor(w, dtype=torch.float32))
    # "sampler" or "none": unweighted CORAL
    return CORALOrdinalFocalLoss(num_classes=5, gamma=gamma, alpha=0.5,
                                 threshold_weights=torch.ones(4, dtype=torch.float32))


def _stage2_loss(tri, y, y_ref, coral_fn, qwk_fn, lw: Dict[str, float], qwk_weight: float) -> torch.Tensor:
    loss = lw.get("ordinal", 1.0) * coral_fn(tri.ordinal_logits, y)
    if qwk_weight > 0:
        loss = loss + qwk_weight * qwk_fn(tri.class_probs, y, is_probs=True)
    loss = loss + lw.get("referable_bce", 0.5) * F.binary_cross_entropy_with_logits(tri.referable_logit.float(), y_ref)
    loss = loss + lw.get("continuous_smooth_l1", 0.4) * F.smooth_l1_loss(tri.continuous_score.float().squeeze(1), y.float())
    return loss


@torch.no_grad()
def predict_stage2(model, loader, accel: ColabTPUAccelerator) -> Dict[str, np.ndarray]:
    model.eval()
    probs, cont, ref, ys, ids = [], [], [], [], []
    for batch in loader:
        x = batch["image"].to(accel.device, non_blocking=True)
        # Evaluate in fp32 like the deployed pipeline: fp16 can overflow on a few images in eval mode.
        with accel.autocast_context(enabled=False):
            tri = model(x)["tri_head"]
        probs.append(tri.class_probs.float().cpu().numpy())
        cont.append(tri.continuous_score.float().squeeze(1).cpu().numpy())
        ref.append(tri.referable_prob.float().squeeze(1).cpu().numpy())
        ys.append(batch["dr_grade"].numpy())
        ids.extend(batch["image_id"])
    probs_arr = np.concatenate(probs)
    if not np.isfinite(probs_arr).all():
        raise RuntimeError(
            f"Model produced NaN/inf for {int((~np.isfinite(probs_arr)).any(1).sum())} images "
            f"(weights finite: {model_is_finite(model)}). Retry with the override 'training.amp=false'."
        )
    return {
        "probs": probs_arr,
        "continuous_scores": np.concatenate(cont),
        "referable_probs": np.concatenate(ref),
        "targets": np.concatenate(ys),
        "image_ids": np.asarray(ids),
    }


def stage2_metrics(pred: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Clinical metrics for argmax grades + QWK of default-thresholded continuous score."""
    y = pred["targets"]
    m = compute_clinical_metrics(y, pred["probs"].argmax(1), pred["probs"])
    m["qwk_continuous_default_thr"] = round(
        quadratic_weighted_kappa(y, NelderMeadQWKOptimizer.discretize(pred["continuous_scores"], DEFAULT_THRESHOLDS)), 4
    )
    return m


def train_stage2_fold(
    cfg: Dict[str, Any],
    val_fold: int,
    data_root: str | Path,
    out_dir: str | Path,
    mirror_dir: Optional[str | Path] = None,
    accel: Optional[ColabTPUAccelerator] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    data_root, out_dir = Path(data_root), Path(out_dir)
    mirror = Path(mirror_dir) if mirror_dir else None
    accel = accel or ColabTPUAccelerator(prefer_tpu=False)
    tcfg, dcfg = cfg["training"], cfg["data"]
    seed = int(tcfg.get("seed", 42)) + val_fold
    set_seed(seed)

    manifest = str(data_root / dcfg.get("manifest", "stage2_grading_splits.csv"))
    aug = RetinalTrainAugment(dcfg.get("augmentation", "medium"))
    phases: List[Dict[str, Any]] = cfg["curriculum"]
    total_epochs = sum(int(p["epochs"]) for p in phases)

    model_spec = {"task": "stage2", **cfg["model"]}
    model = build_model(model_spec).to(accel.device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, float(tcfg.get("backbone_lr_mult", 0.2)), float(tcfg["weight_decay"])),
        lr=float(phases[0]["lr"]),
    )
    ema = ModelEMA(model, decay=float(tcfg["ema_decay"])) if tcfg.get("ema_decay") else None
    accel.set_amp(bool(tcfg.get("amp", True)))
    guard = NonFiniteGuard(model)
    qwk_fn = SoftQWKLoss(num_classes=5)
    lw = cfg.get("loss", {}).get("weights", {})
    strategy = cfg.get("imbalance", {}).get("strategy", "sampler")
    select_metric = tcfg.get("select_metric", "qwk")

    last_path = out_dir / f"stage2_fold{val_fold}_last.pt"
    best_path = out_dir / fold_ckpt_name(val_fold)
    start_epoch, best_score, history = 1, -1.0, []
    if resume and (state := load_training_state(last_path)) is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if ema and state.get("ema"):
            ema.load_state_dict(state["ema"])
        if accel.grad_scaler is not None and state.get("scaler"):
            accel.grad_scaler.load_state_dict(state["scaler"])
        start_epoch, best_score, history = state["epoch"] + 1, state["best_score"], state["history"]
        if not model_is_finite(model) or (ema and not model_is_finite(ema.module)):
            raise RuntimeError(
                f"{last_path} contains NaN/inf weights or BatchNorm stats. Delete it (locally and on Drive) to restart."
            )
        print(f"[stage2 fold {val_fold}] resumed from epoch {state['epoch']} (best {select_metric} {best_score:.4f})")

    pin, nw = accel.info.backend == "cuda", int(tcfg.get("num_workers", 2))
    loaders_cache: Dict[int, Any] = {}

    def loaders_for(res: int, bs: int):
        if res not in loaders_cache:
            common = dict(manifest_csv=manifest, resolution=res, data_root_override=str(data_root))
            tr = PreprocessedManifestDataset(split="train_val", exclude_fold=val_fold, augment=aug, **common)
            va = PreprocessedManifestDataset(split="train_val", only_fold=val_fold, **common)
            maybe_subsample(tr, dcfg.get("max_train_samples"), seed)
            maybe_subsample(va, dcfg.get("max_eval_samples"), seed)
            sampler = (
                balanced_sampler(tr.dr_grades(), 5, float(cfg["imbalance"].get("effective_number_beta", 0.999)), seed)
                if strategy == "sampler" else None
            )
            loaders_cache[res] = (
                tr,
                make_loader(tr, bs, True, nw, pin, seed, sampler=sampler, drop_last=True),
                make_loader(va, bs, False, nw, pin, seed),
            )
        return loaders_cache[res]

    epoch_offset = 0
    for phase in phases:
        res, bs = int(phase["resolution"]), int(phase["batch_size"])
        accum = int(phase.get("grad_accum_steps", 1))
        p_epochs = int(phase["epochs"])
        p_start, p_end = epoch_offset + 1, epoch_offset + p_epochs
        epoch_offset = p_end
        if start_epoch > p_end:
            continue
        train_ds, train_loader, val_loader = loaders_for(res, bs)
        coral_fn = _build_coral_loss(cfg, train_ds.dr_grades())
        if hasattr(model, "gem_pool"):
            model.gem_pool.p.requires_grad_(bool(phase.get("train_gem_p", True)))
        steps_per_epoch = max(1, len(train_loader) // accum)
        p_total = p_epochs * steps_per_epoch
        p_warm = int(float(phase.get("warmup_epochs", 0)) * steps_per_epoch)

        for epoch in range(max(start_epoch, p_start), p_end + 1):
            t0 = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            # Soft-QWK weight ramps linearly inside the phase from qwk_start to qwk_end.
            frac = (epoch - p_start) / max(1, p_epochs - 1)
            qwk_w = float(phase.get("soft_qwk_start", 0.0)) + frac * (
                float(phase.get("soft_qwk_end", 0.0)) - float(phase.get("soft_qwk_start", 0.0))
            )
            running, n_b, skipped = 0.0, 0, 0
            for i, batch in enumerate(train_loader):
                step = (epoch - p_start) * steps_per_epoch + i // accum
                set_lr(optimizer, float(phase["lr"]),
                       warmup_cosine_factor(step, p_total, p_warm, float(tcfg.get("min_lr_ratio", 0.02))))
                x = batch["image"].to(accel.device, non_blocking=True)
                y = batch["dr_grade"].to(accel.device)
                y_ref = batch["referable_dr"].to(accel.device)
                guard.before_forward()
                with accel.autocast_context(enabled=True):
                    tri = model(x)["tri_head"]
                loss = _stage2_loss(tri, y, y_ref, coral_fn, qwk_fn, lw, qwk_w) / accum
                if not guard.check(loss):
                    optimizer.zero_grad(set_to_none=True)  # drop this (and any partially accumulated) batch
                    continue
                accel.backward(loss)
                if (i + 1) % accum == 0:
                    if not accel.clip_and_step(optimizer, trainable_parameters(model), tcfg.get("grad_clip", 2.0)):
                        skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    if ema:
                        ema.update(model)
                running += float(loss.item()) * accum
                n_b += 1

            skipped_bad = guard.end_epoch(f"[stage2 fold {val_fold}] epoch {epoch}")
            eval_model = ema.module if ema else model
            val_pred = predict_stage2(eval_model, val_loader, accel)
            vm = stage2_metrics(val_pred)
            score = vm.get(select_metric, float("nan"))
            improved = bool(np.isfinite(score) and score > best_score)
            if improved:
                best_score = float(score)
                save_model_checkpoint(
                    best_path, eval_model, {**model_spec, "input_size": res},
                    {"epoch": epoch, "val_fold": val_fold, "phase": phase.get("name", ""), "val_metrics": vm},
                )
                np.savez_compressed(out_dir / f"stage2_fold{val_fold}_val_preds.npz", **val_pred)
                mirror_to(best_path, mirror)
                mirror_to(out_dir / f"stage2_fold{val_fold}_val_preds.npz", mirror)

            history.append({"epoch": epoch, "phase": phase.get("name", ""), "resolution": res,
                            "train_loss": running / max(n_b, 1), "soft_qwk_weight": qwk_w, **vm})
            save_training_state(
                last_path,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                ema=ema.state_dict() if ema else None,
                scaler=accel.grad_scaler.state_dict() if accel.grad_scaler is not None else None,
                epoch=epoch, best_score=best_score, history=history,
            )
            mirror_to(last_path, mirror)
            print(
                f"[stage2 fold {val_fold}] epoch {epoch:02d}/{total_epochs} {phase.get('name', '')}@{res}px "
                f"({time.time() - t0:.0f}s) loss={running / max(n_b, 1):.4f} QWK={vm['qwk']:.4f} "
                f"QWK(cont)={vm['qwk_continuous_default_thr']:.4f} F1={vm['macro_f1']:.4f} "
                f"mildRecall={vm['mild_dr_recall']:.3f} refAUC={vm['referable_auc']:.4f}"
                + (f" skipped_steps={skipped}" if skipped else "")
                + (f" nonfinite_batches={skipped_bad}" if skipped_bad else "")
                + (" *saved*" if improved else "")
            )

    # Test-split predictions of this fold's best model (at its own resolution).
    best_model, best_spec, _ = load_model_checkpoint(best_path, device=accel.device)
    test_ds = PreprocessedManifestDataset(
        manifest_csv=manifest, split="test", resolution=int(best_spec["input_size"]), data_root_override=str(data_root)
    )
    # Same subset for every fold so fold test predictions stay aligned for ensembling.
    maybe_subsample(test_ds, dcfg.get("max_eval_samples"), int(tcfg.get("seed", 42)))
    bs_eval = int(phases[-1]["batch_size"])
    test_pred = predict_stage2(best_model, make_loader(test_ds, bs_eval, False, nw, pin, seed), accel)
    np.savez_compressed(out_dir / f"stage2_fold{val_fold}_test_preds.npz", **test_pred)
    mirror_to(out_dir / f"stage2_fold{val_fold}_test_preds.npz", mirror)
    summary = {"val_fold": val_fold, "best_val_score": best_score, "test_metrics": stage2_metrics(test_pred), "history": history}
    write_json(out_dir / f"stage2_fold{val_fold}_metrics.json", summary)
    (out_dir / f"stage2_fold{val_fold}.done").write_text("done\n")
    mirror_to(out_dir / f"stage2_fold{val_fold}_metrics.json", mirror)
    mirror_to(out_dir / f"stage2_fold{val_fold}.done", mirror)
    return summary


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def aggregate_cv(out_dir: str | Path, folds: List[int], mirror_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Concatenate OOF predictions and average fold models' test predictions."""
    out_dir = Path(out_dir)
    mirror = Path(mirror_dir) if mirror_dir else None
    oof_parts, test_parts = [], []
    for f in folds:
        v = _load_npz(out_dir / f"stage2_fold{f}_val_preds.npz")
        v["fold"] = np.full(len(v["targets"]), f, dtype=np.int64)
        oof_parts.append(v)
        test_parts.append(_load_npz(out_dir / f"stage2_fold{f}_test_preds.npz"))

    oof = {k: np.concatenate([p[k] for p in oof_parts]) for k in oof_parts[0]}
    np.savez_compressed(out_dir / "stage2_oof.npz", **oof)

    ref_ids = test_parts[0]["image_ids"]
    for p in test_parts[1:]:
        if not np.array_equal(p["image_ids"], ref_ids):
            raise RuntimeError("Fold test predictions are not aligned by image_id.")
    test_ens = {
        "probs": np.mean([p["probs"] for p in test_parts], axis=0),
        "continuous_scores": np.mean([p["continuous_scores"] for p in test_parts], axis=0),
        "referable_probs": np.mean([p["referable_probs"] for p in test_parts], axis=0),
        "member_referable_std": np.std([p["referable_probs"] for p in test_parts], axis=0),
        "targets": test_parts[0]["targets"],
        "image_ids": ref_ids,
    }
    np.savez_compressed(out_dir / "stage2_test_ensemble.npz", **test_ens)

    summary = {
        "folds": folds,
        "oof_metrics": stage2_metrics(oof),
        "test_ensemble_metrics": stage2_metrics(test_ens),
        "test_per_fold_metrics": {f: stage2_metrics(p) for f, p in zip(folds, test_parts)},
    }
    write_json(out_dir / "stage2_cv_metrics.json", summary)
    for name in ("stage2_oof.npz", "stage2_test_ensemble.npz", "stage2_cv_metrics.json"):
        mirror_to(out_dir / name, mirror)
    return summary


def run_stage2_cv(
    cfg: Dict[str, Any],
    data_root: str | Path,
    out_dir: str | Path,
    mirror_dir: Optional[str | Path] = None,
    folds: Optional[List[int]] = None,
    accel: Optional[ColabTPUAccelerator] = None,
) -> Optional[Dict[str, Any]]:
    """Train the requested folds (skipping finished ones).

    Aggregation (OOF + test ensemble) runs once *all* folds listed in `cfg["cv"]["folds"]`
    are finished, so `folds` can be a subset trained in one Colab session.
    """
    out_dir = Path(out_dir)
    all_folds = list(cfg.get("cv", {}).get("folds", [1, 2, 3, 4]))
    folds = list(folds or all_folds)
    accel = accel or ColabTPUAccelerator(prefer_tpu=False)
    mirror = Path(mirror_dir) if mirror_dir else None
    for f in sorted(set(folds) | set(all_folds)):
        # Restore finished-fold artifacts from the mirror (e.g. Drive) after a new Colab session.
        if mirror is not None and (mirror / f"stage2_fold{f}.done").exists() and not (out_dir / f"stage2_fold{f}.done").exists():
            for name in (fold_ckpt_name(f), f"stage2_fold{f}_val_preds.npz", f"stage2_fold{f}_test_preds.npz",
                         f"stage2_fold{f}_metrics.json", f"stage2_fold{f}.done"):
                if (mirror / name).exists():
                    mirror_to(mirror / name, out_dir)
        if (out_dir / f"stage2_fold{f}.done").exists():
            if f in folds:
                print(f"[stage2] fold {f} already finished - skipping")
            continue
        if f not in folds:
            continue
        if mirror is not None and (mirror / f"stage2_fold{f}_last.pt").exists() and not (out_dir / f"stage2_fold{f}_last.pt").exists():
            mirror_to(mirror / f"stage2_fold{f}_last.pt", out_dir)
        train_stage2_fold(cfg, f, data_root, out_dir, mirror_dir, accel)

    if all((out_dir / f"stage2_fold{f}.done").exists() for f in all_folds):
        summary = aggregate_cv(out_dir, all_folds, mirror_dir)
        print("[stage2] OOF:", summary["oof_metrics"])
        print("[stage2] TEST (fold ensemble):", summary["test_ensemble_metrics"])
        return summary
    missing = [f for f in all_folds if not (out_dir / f"stage2_fold{f}.done").exists()]
    print(f"[stage2] folds still to train: {missing}")
    return None
