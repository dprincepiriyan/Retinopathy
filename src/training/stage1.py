"""Stage-1 multi-label triage training (DR / glaucoma / cataract).

Driven by `configs/tier2_stage1_triage.yaml`. Resumable: a `stage1_last.pt` training
state is written every epoch, so a Colab disconnect only loses the current epoch.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data.datasets import PreprocessedManifestDataset
from src.data.augmentations import RetinalTrainAugment
from src.data.stage1_manifest import ensure_stage1_combined_manifest
from src.losses import AsymmetricFocalLoss
from src.models.factory import build_model, save_model_checkpoint
from src.models.pretrained import parameter_groups
from src.training.common import (
    ModelEMA,
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
from src.utils.metrics import multilabel_triage_metrics
from src.utils.tpu_accelerator import ColabTPUAccelerator

BEST_CKPT_NAME = "stage1_multi_label_best.pth"
LAST_STATE_NAME = "stage1_last.pt"


def _stage1_loss(criterion, out: Dict[str, Any], y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    total = 0.0
    for w, logits in zip(out["member_weights"], out["member_logits"]):
        total = total + float(w) * criterion(logits, y, mask)
    return total


@torch.no_grad()
def evaluate_stage1(model, loader, criterion, accel: ColabTPUAccelerator) -> Dict[str, Any]:
    model.eval()
    probs, targets, masks, sources, ids = [], [], [], [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(accel.device, non_blocking=True)
        y = batch["multi_label"].to(accel.device)
        m = batch["multi_label_mask"].to(accel.device)
        with accel.autocast_context(enabled=True):
            out = model(x)
        loss = _stage1_loss(criterion, out, y, m)
        total_loss += float(loss.item())
        n += 1
        probs.append(out["probs"].float().cpu().numpy())
        targets.append(y.cpu().numpy())
        masks.append(m.cpu().numpy())
        sources.extend(batch["source_dataset"])
        ids.extend(batch["image_id"])

    probs_arr = np.concatenate(probs)
    targets_arr = np.concatenate(targets)
    masks_arr = np.concatenate(masks)
    metrics = multilabel_triage_metrics(probs_arr, targets_arr, masks_arr, sources)
    metrics["loss"] = round(total_loss / max(n, 1), 4)
    return {
        "metrics": metrics,
        "probs": probs_arr,
        "targets": targets_arr,
        "masks": masks_arr,
        "sources": np.asarray(sources),
        "image_ids": np.asarray(ids),
    }


def train_stage1(
    cfg: Dict[str, Any],
    data_root: str | Path,
    out_dir: str | Path,
    mirror_dir: Optional[str | Path] = None,
    accel: Optional[ColabTPUAccelerator] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Train the Stage-1 triage model and evaluate the best checkpoint on the locked test split."""
    data_root, out_dir = Path(data_root), Path(out_dir)
    mirror = Path(mirror_dir) if mirror_dir else None
    accel = accel or ColabTPUAccelerator(prefer_tpu=False)
    tcfg, dcfg = cfg["training"], cfg["data"]
    seed = int(tcfg.get("seed", 42))
    set_seed(seed)

    manifest = (
        ensure_stage1_combined_manifest(data_root, include_aptos=dcfg.get("include_aptos", True))
        if dcfg.get("use_combined_manifest", True)
        else data_root / "stage1_triage_splits.csv"
    )
    res = int(dcfg.get("resolution", 512))
    val_fold = int(dcfg.get("val_fold", 4))
    aug = RetinalTrainAugment(dcfg.get("augmentation", "medium"))
    common = dict(manifest_csv=str(manifest), resolution=res, data_root_override=str(data_root))
    train_ds = PreprocessedManifestDataset(split="train_val", exclude_fold=val_fold, augment=aug, **common)
    val_ds = PreprocessedManifestDataset(split="train_val", only_fold=val_fold, **common)
    test_ds = PreprocessedManifestDataset(split="test", **common)
    maybe_subsample(train_ds, dcfg.get("max_train_samples"), seed)
    maybe_subsample(val_ds, dcfg.get("max_eval_samples"), seed)
    maybe_subsample(test_ds, dcfg.get("max_eval_samples"), seed)

    bs, nw = int(tcfg["batch_size"]), int(tcfg.get("num_workers", 2))
    pin = accel.info.backend == "cuda"
    train_loader = make_loader(train_ds, bs, True, nw, pin, seed, drop_last=True)
    val_loader = make_loader(val_ds, bs, False, nw, pin, seed)
    test_loader = make_loader(test_ds, bs, False, nw, pin, seed)
    print(f"[stage1] manifest={manifest.name} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    spec = {"task": "stage1", **cfg["model"], "input_size": res}
    model = build_model(spec).to(accel.device)
    lcfg = cfg.get("loss", {})
    criterion = AsymmetricFocalLoss(
        gamma_neg=lcfg.get("gamma_neg", 4.0),
        gamma_pos=lcfg.get("gamma_pos", 1.0),
        clip=lcfg.get("clip", 0.05),
        label_smoothing=lcfg.get("label_smoothing", 0.0),
    )
    optimizer = torch.optim.AdamW(
        parameter_groups(model, float(tcfg.get("backbone_lr_mult", 0.2)), float(tcfg["weight_decay"])),
        lr=float(tcfg["lr"]),
    )
    ema = ModelEMA(model, decay=float(tcfg["ema_decay"])) if tcfg.get("ema_decay") else None

    epochs = int(tcfg["epochs"])
    accum = int(tcfg.get("grad_accum_steps", 1))
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(float(tcfg.get("warmup_epochs", 1)) * steps_per_epoch)
    min_factor = float(tcfg.get("min_lr_ratio", 0.01))

    start_epoch, best_score, history = 1, -1.0, []
    last_path = out_dir / LAST_STATE_NAME
    if resume and mirror is not None and not last_path.exists():
        # New Colab session: pull the resumable state (and best weights) back from Drive.
        for name in (LAST_STATE_NAME, BEST_CKPT_NAME):
            if (mirror / name).exists():
                mirror_to(mirror / name, out_dir)
    if resume and (state := load_training_state(last_path)) is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if ema and state.get("ema"):
            ema.load_state_dict(state["ema"])
        if accel.grad_scaler is not None and state.get("scaler"):
            accel.grad_scaler.load_state_dict(state["scaler"])
        start_epoch, best_score, history = state["epoch"] + 1, state["best_score"], state["history"]
        print(f"[stage1] resumed from epoch {state['epoch']} (best macro-AUC {best_score:.4f})")

    best_path = out_dir / BEST_CKPT_NAME
    select_metric = tcfg.get("select_metric", "macro_auc")
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, n_batches, skipped = 0.0, 0, 0
        for i, batch in enumerate(train_loader):
            step = (epoch - 1) * steps_per_epoch + i // accum
            set_lr(optimizer, float(tcfg["lr"]), warmup_cosine_factor(step, total_steps, warmup_steps, min_factor))
            x = batch["image"].to(accel.device, non_blocking=True)
            y = batch["multi_label"].to(accel.device)
            m = batch["multi_label_mask"].to(accel.device)
            with accel.autocast_context(enabled=True):
                out = model(x)
            loss = _stage1_loss(criterion, out, y, m) / accum
            accel.backward(loss)
            if (i + 1) % accum == 0:
                if not accel.clip_and_step(optimizer, trainable_parameters(model), tcfg.get("grad_clip", 2.0)):
                    skipped += 1
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
            running += float(loss.item()) * accum
            n_batches += 1

        eval_model = ema.module if ema else model
        val = evaluate_stage1(eval_model, val_loader, criterion, accel)
        vm = val["metrics"]
        score = vm.get(select_metric, float("nan"))
        improved = bool(np.isfinite(score) and score > best_score)
        if improved:
            best_score = float(score)
            save_model_checkpoint(best_path, eval_model, spec, {"epoch": epoch, "val_metrics": vm, "val_fold": val_fold})
            np.savez_compressed(
                out_dir / "stage1_val_preds.npz",
                probs=val["probs"], targets=val["targets"], masks=val["masks"],
                sources=val["sources"], image_ids=val["image_ids"],
            )
            mirror_to(best_path, mirror)
            mirror_to(out_dir / "stage1_val_preds.npz", mirror)

        history.append({"epoch": epoch, "train_loss": running / max(n_batches, 1), **vm})
        save_training_state(
            last_path,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            ema=ema.state_dict() if ema else None,
            scaler=accel.grad_scaler.state_dict() if accel.grad_scaler is not None else None,
            epoch=epoch,
            best_score=best_score,
            history=history,
        )
        mirror_to(last_path, mirror)
        print(
            f"[stage1] epoch {epoch:02d}/{epochs} ({time.time() - t0:.0f}s) "
            f"train_loss={running / max(n_batches, 1):.4f} val_loss={vm['loss']:.4f} "
            f"macroAUC={vm['macro_auc']:.4f} DR={vm.get('auc_dr', float('nan')):.3f} "
            f"DR(within-source)={vm.get('auc_dr_within_source_mean', float('nan')):.3f} "
            f"GL={vm.get('auc_glaucoma', float('nan')):.3f} CAT={vm.get('auc_cataract', float('nan')):.3f}"
            + (f" skipped_steps={skipped}" if skipped else "")
            + (" *saved*" if improved else "")
        )

    # Final, single evaluation of the best checkpoint on the locked test split.
    from src.models.factory import load_model_checkpoint

    best_model, _, _ = load_model_checkpoint(best_path, device=accel.device)
    test = evaluate_stage1(best_model, test_loader, criterion, accel)
    summary = {"best_val_score": best_score, "select_metric": select_metric, "test_metrics": test["metrics"], "history": history}
    write_json(out_dir / "stage1_metrics.json", summary)
    mirror_to(out_dir / "stage1_metrics.json", mirror)
    print("[stage1] TEST:", test["metrics"])
    return summary
