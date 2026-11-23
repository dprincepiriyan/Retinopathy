"""Shared training utilities: seeding, EMA, LR schedule, resumable state, samplers."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.samplers import EffectiveNumberSampler


# --------------------------------------------------------------------------- seeding
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:  # pragma: no cover - runs inside DataLoader workers
    """Give each DataLoader worker a distinct, reproducible NumPy/random seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------- data
def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Default collate that keeps string fields as lists."""
    out: Dict[str, Any] = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        out[key] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    sampler=None,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        collate_fn=collate_batch,
        persistent_workers=num_workers > 0,
    )


def balanced_sampler(labels: np.ndarray, num_classes: int, beta: float, seed: int) -> WeightedRandomSampler:
    """Class-balanced (effective-number) sampler over integer labels."""
    probs = EffectiveNumberSampler(beta=beta).compute_sample_probabilities(labels, num_classes=num_classes)
    return WeightedRandomSampler(
        weights=torch.from_numpy(probs).double(),
        num_samples=len(labels),
        replacement=True,
        generator=make_generator(seed),
    )


# --------------------------------------------------------------------------- schedule
def warmup_cosine_factor(step: int, total_steps: int, warmup_steps: int, min_factor: float) -> float:
    """LR multiplier: linear warmup to 1.0 then cosine decay to `min_factor`."""
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-3, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, base_lr: float, factor: float) -> float:
    lr = base_lr * factor
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_mult", 1.0)
    return lr


# --------------------------------------------------------------------------- EMA
class ModelEMA:
    """Exponential moving average of model weights (evaluated instead of raw weights)."""

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 200) -> None:
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # Ramp decay up early so EMA is not dominated by random init weights.
        d = min(self.decay, (1.0 + self.updates) / (self.warmup_steps + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(src)

    def state_dict(self) -> Dict[str, Any]:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = int(state.get("updates", 0))


# --------------------------------------------------------------------------- resumable state
def save_training_state(path: Path, **state: Any) -> None:
    """Atomically save everything needed to resume after a Colab disconnect."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, str(tmp))
    tmp.replace(path)


def load_training_state(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return torch.load(str(path), map_location="cpu", weights_only=False)


def mirror_to(src: Path, dst_dir: Optional[Path]) -> None:
    """Copy a file into a mirror directory (e.g. Google Drive) if it differs from the source dir."""
    if dst_dir is None:
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.resolve() != src.resolve():
        shutil.copy2(src, dst)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def maybe_subsample(dataset, max_samples: Optional[int], seed: int):
    """Randomly keep at most `max_samples` rows (for smoke tests / quick debugging)."""
    if max_samples and len(dataset.rows) > int(max_samples):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(dataset.rows), size=int(max_samples), replace=False))
        dataset.rows = [dataset.rows[i] for i in keep]
    return dataset


# --------------------------------------------------------------------------- fp16 safety
def model_is_finite(model: nn.Module) -> bool:
    """True if every parameter and floating buffer (incl. BatchNorm running stats) is finite."""
    tensors = [p.detach() for p in model.parameters()]
    tensors += [b for b in model.buffers() if b.dtype.is_floating_point]
    return bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all())


class NonFiniteGuard:
    """Protects training from fp16 overflow batches.

    GradScaler already skips the *optimizer* step when gradients overflow, but the
    forward pass of that batch has by then written inf/NaN into BatchNorm running
    statistics. Training (batch statistics) keeps working, while every evaluation
    (running statistics) returns NaN. This guard snapshots the running stats before
    each forward pass and restores them when the loss or the stats become non-finite,
    and the batch is skipped.
    """

    def __init__(self, model: nn.Module, max_bad_fraction: float = 0.2) -> None:
        self.buffers = [
            b for n, b in model.named_buffers() if n.endswith(("running_mean", "running_var"))
        ]
        self.max_bad_fraction = max_bad_fraction
        self._snapshot: List[torch.Tensor] = []
        self.bad = 0
        self.total = 0

    def before_forward(self) -> None:
        self._snapshot = [b.detach().clone() for b in self.buffers]

    def check(self, loss: torch.Tensor) -> bool:
        """Return True if the batch is usable; otherwise restore BN stats and return False."""
        self.total += 1
        flags = [torch.isfinite(loss.detach()).all()] + [torch.isfinite(b).all() for b in self.buffers]
        if bool(torch.stack(flags).all()):
            return True
        with torch.no_grad():
            for buf, snap in zip(self.buffers, self._snapshot):
                buf.copy_(snap)
        self.bad += 1
        return False

    def end_epoch(self, tag: str) -> int:
        bad, total = self.bad, self.total
        self.bad, self.total = 0, 0
        if total and bad / total > self.max_bad_fraction:
            raise RuntimeError(
                f"{tag}: {bad}/{total} training batches produced NaN/inf. Mixed precision is unstable "
                "for this run; disable it with the config override 'training.amp=false'."
            )
        return bad
