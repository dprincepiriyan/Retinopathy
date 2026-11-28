"""CUDA / CPU accelerator abstraction for training and evaluation.

IMPORTANT: Colab T4 GPUs have no native bfloat16, so CUDA autocast uses float16 there.
float16 training REQUIRES loss scaling, otherwise small gradients underflow to zero.
`backward()` / `clip_and_step()` below apply a `torch.amp.GradScaler` automatically
whenever float16 autocast is active.

(The module/class names keep "TPU" only so existing notebooks keep importing them;
TPU support was removed because every caller uses a GPU runtime.)
"""

from contextlib import nullcontext
from dataclasses import dataclass
import torch


@dataclass
class AcceleratorInfo:
    """Runtime accelerator metadata."""

    backend: str          # "cuda" or "cpu"
    device: torch.device
    amp_dtype: torch.dtype
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
    """Picks CUDA when available (else CPU) and handles mixed precision + loss scaling."""

    def __init__(self, prefer_tpu: bool = False) -> None:
        # `prefer_tpu` is ignored; kept so notebooks calling ColabTPUAccelerator(prefer_tpu=False) still work.
        self.info = self._detect_hardware()
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

    @staticmethod
    def _detect_hardware() -> AcceleratorInfo:
        if torch.cuda.is_available():
            return AcceleratorInfo(
                backend="cuda",
                device=torch.device("cuda"),
                amp_dtype=torch.bfloat16 if cuda_has_native_bf16() else torch.float16,
                device_name=torch.cuda.get_device_name(0),
            )
        return AcceleratorInfo(backend="cpu", device=torch.device("cpu"), amp_dtype=torch.bfloat16, device_name="CPU")

    @property
    def device(self) -> torch.device:
        return self.info.device

    def autocast_context(self, enabled: bool = True):
        """Return the mixed-precision autocast context (no-op on CPU or when AMP is off)."""
        if not enabled or not getattr(self, "amp_enabled", True) or self.info.backend != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype=self.info.amp_dtype)

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
        optimizer.step()
        return True
