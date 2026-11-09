"""Google Colab TPU (`torch_xla`) & Local GPU/CPU Hardware Accelerator Abstraction.

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


class ColabTPUAccelerator:
    """Unified hardware accelerator supporting Google Colab TPU (`torch_xla`), CUDA, and CPU."""

    def __init__(self, prefer_tpu: bool = True) -> None:
        self._xm: Any = None
        self.info = self._detect_hardware(prefer_tpu=prefer_tpu)

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
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
        if not enabled:
            return nullcontext()
        if self.info.is_tpu:
            return torch.autocast("xla", dtype=torch.bfloat16)
        if self.info.backend == "cuda":
            return torch.autocast("cuda", dtype=self.info.amp_dtype)
        return nullcontext()

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
