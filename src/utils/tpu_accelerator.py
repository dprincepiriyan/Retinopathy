"""Google Colab TPU (`torch_xla`) & Local GPU/CPU Hardware Accelerator Abstraction.

IMPORTANT: Colab T4 GPUs have no bfloat16 support, so CUDA autocast falls back to
float16. float16 training REQUIRES loss scaling, otherwise small gradients underflow
to zero. `backward()` / `clip_and_step()` below apply a `torch.amp.GradScaler`
automatically whenever float16 autocast is active. GPU runtimes (T4/L4/A100) are
recommended over TPU for this project (dynamic resolutions + small batches).

Optimizes RetinaNet-Pro training specifically for Google Colab TPUs (v2-8 / v5e-1):
1. Native `bfloat16` mixed precision on TPU (avoids FP16 GradScaler XLA graph breaks).
2. Enforces static batch/spatial dimensions (`drop_last=True`) to prevent expensive
   XLA dynamic-shape recompilations.
3. Wraps `xm.optimizer_step(optimizer)` and `xm.mark_step()` transparently while
   remaining 100% compatible with local CUDA GPUs (RTX 3050) and CPUs.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn


@dataclass
class AcceleratorInfo:
    """Runtime accelerator metadata."""

    backend: str          # "tpu", "cuda", or "cpu"
    device: torch.device
    amp_dtype: torch.dtype
    is_tpu: bool
    device_name: str


def cuda_has_native_bf16() -> bool:
    """True only for GPUs with *native* bfloat16 (Ampere / compute capability >= 8.0).

    `torch.cuda.is_bf16_supported()` also returns True on T4 / V100 because bf16 can be
    emulated there, but emulated bf16 is far slower than native fp16 + GradScaler.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


class ColabTPUAccelerator:
    """Unified hardware accelerator supporting Google Colab TPU (`torch_xla`), CUDA, and CPU."""

    def __init__(self, prefer_tpu: bool = False) -> None:
        self._xm: Any = None
        self.info = self._detect_hardware(prefer_tpu=prefer_tpu)
        self.amp_enabled = True
        self.grad_scaler = None
        if self.uses_grad_scaler:
            self.grad_scaler = torch.amp.GradScaler("cuda")

    def set_amp(self, enabled: bool) -> None:
        """Turn mixed precision on/off (off = plain fp32: slower but immune to fp16 overflow)."""
        self.amp_enabled = bool(enabled)
        if not self.amp_enabled:
            self.grad_scaler = None
        elif self.uses_grad_scaler and self.grad_scaler is None:
            self.grad_scaler = torch.amp.GradScaler("cuda")

    @property
    def uses_grad_scaler(self) -> bool:
        """float16 autocast on CUDA needs dynamic loss scaling."""
        return (
            self.info.backend == "cuda"
            and self.info.amp_dtype == torch.float16
            and getattr(self, "amp_enabled", True)
        )

    def _detect_hardware(self, prefer_tpu: bool = True) -> AcceleratorInfo:
        if prefer_tpu:
            try:
                import torch_xla.core.xla_model as xm  # type: ignore

                self._xm = xm
                dev = xm.xla_device()
                return AcceleratorInfo(
                    backend="tpu",
                    device=dev,
                    amp_dtype=torch.bfloat16,
                    is_tpu=True,
                    device_name=f"Google Colab TPU ({dev})",
                )
            except ImportError:
                pass

        if torch.cuda.is_available():
            dev = torch.device("cuda")
            amp_dtype = torch.bfloat16 if cuda_has_native_bf16() else torch.float16
            return AcceleratorInfo(
                backend="cuda",
                device=dev,
                amp_dtype=amp_dtype,
                is_tpu=False,
                device_name=torch.cuda.get_device_name(0),
            )

        return AcceleratorInfo(
            backend="cpu",
            device=torch.device("cpu"),
            amp_dtype=torch.bfloat16,
            is_tpu=False,
            device_name="CPU",
        )

    @property
    def device(self) -> torch.device:
        return self.info.device

    def autocast_context(self, enabled: bool = True):
        """Return the appropriate mixed-precision autocast context manager (`bfloat16` on TPU)."""
        if not enabled or not getattr(self, "amp_enabled", True):
            return nullcontext()
        if self.info.is_tpu:
            return torch.autocast("xla", dtype=torch.bfloat16)
        if self.info.backend == "cuda":
            return torch.autocast("cuda", dtype=self.info.amp_dtype)
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with loss scaling when float16 autocast is active."""
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_and_step(
        self,
        optimizer: torch.optim.Optimizer,
        parameters,
        max_norm: float | None = None,
    ) -> bool:
        """Unscale (if needed), clip gradients, step the optimizer and update the scaler.

        Returns False when the step was skipped because of inf/NaN gradients (fp16 overflow).
        """
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(optimizer)
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
            scale_before = self.grad_scaler.get_scale()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
            return self.grad_scaler.get_scale() >= scale_before
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
        self.optimizer_step(optimizer)
        return True

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Execute optimizer step and synchronize XLA graph on TPU."""
        if self.info.is_tpu and self._xm is not None:
            self._xm.optimizer_step(optimizer, barrier=True)
            self._xm.mark_step()
        else:
            optimizer.step()

    def save_checkpoint(self, model: nn.Module, path: str | Path) -> Path:
        """Save model state_dict moved safely to CPU so checkpoints load on any laptop GPU/CPU."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if self.info.is_tpu and self._xm is not None:
            self._xm.save(cpu_state, str(out_path))
        else:
            torch.save(cpu_state, str(out_path))
        return out_path
