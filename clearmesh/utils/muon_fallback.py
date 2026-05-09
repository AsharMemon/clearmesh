"""Small Muon optimizer fallback for environments before ``torch.optim.Muon``.

The FACE paper trains with Muon. Thunder's current CUDA image can lag behind
the PyTorch release that ships native Muon, so this fallback keeps the research
script on the same optimizer family instead of silently switching to AdamW for
hidden-layer matrices.
"""

from __future__ import annotations

import math
from typing import Iterable


def build_muon_fallback(torch):  # type: ignore[no-untyped-def]
    class MuonFallback(torch.optim.Optimizer):
        def __init__(
            self,
            params: Iterable,
            *,
            lr: float = 1e-3,
            weight_decay: float = 0.1,
            momentum: float = 0.95,
            nesterov: bool = True,
            ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
            eps: float = 1e-7,
            ns_steps: int = 5,
        ) -> None:
            defaults = {
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_coefficients": ns_coefficients,
                "eps": eps,
                "ns_steps": ns_steps,
            }
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):  # type: ignore[no-untyped-def]
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = float(group["lr"])
                weight_decay = float(group["weight_decay"])
                momentum = float(group["momentum"])
                nesterov = bool(group["nesterov"])
                ns_coefficients = tuple(group["ns_coefficients"])
                eps = float(group["eps"])
                ns_steps = int(group["ns_steps"])
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if param.ndim != 2:
                        raise ValueError(f"MuonFallback only supports 2D parameters, got {tuple(param.shape)}")
                    grad = param.grad
                    state = self.state[param]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    buffer = state["momentum_buffer"]
                    buffer.mul_(momentum).add_(grad)
                    update = grad.add(buffer, alpha=momentum) if nesterov else buffer
                    update = _newton_schulz_orthogonalize(
                        torch,
                        update,
                        ns_coefficients=ns_coefficients,
                        eps=eps,
                        steps=ns_steps,
                    )
                    rows, cols = int(param.shape[0]), int(param.shape[1])
                    adjusted_lr = lr * math.sqrt(max(1.0, rows / max(1, cols)))
                    if weight_decay:
                        param.add_(param, alpha=-lr * weight_decay)
                    param.add_(update.to(dtype=param.dtype), alpha=-adjusted_lr)
            return loss

    return MuonFallback


def _newton_schulz_orthogonalize(
    torch,  # type: ignore[no-untyped-def]
    matrix,
    *,
    ns_coefficients: tuple[float, float, float],
    eps: float,
    steps: int,
):  # type: ignore[no-untyped-def]
    a, b, c = ns_coefficients
    original_dtype = matrix.dtype
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (torch.linalg.vector_norm(x) + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)
