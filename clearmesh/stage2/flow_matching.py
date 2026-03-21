"""Flow matching utilities for Stage 2 v2 refinement DiT.

Implements rectified flow (linear interpolation path) matching TRELLIS.2's
native training formulation. Convention: t=0 is data, t=1 is noise.

Flow matching training:
    x_0 = fine_slat (data)
    x_1 = noise ~ N(0, I)
    t ~ U(0, 1)
    x_t = (1-t) * x_0 + t * x_1
    v_target = x_1 - x_0
    v_pred = model(x_t, coarse_slat, positions, t, cond)
    loss = MSE(v_pred, v_target)

Flow matching inference (50 Euler steps from t=1 → t=0):
    sigmas = linspace(1, 0, 51)
    for each step: x_t += dt * v_pred
"""

import torch


def sample_timestep(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample t ~ U(0, 1) for flow matching training."""
    return torch.rand(batch_size, device=device)


def interpolate(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation: x_t = (1-t)*x_0 + t*x_1.

    Args:
        x_0: (B, N, D) data (fine SLAT)
        x_1: (B, N, D) noise
        t: (B,) timesteps in [0, 1]
    """
    t = t[:, None, None]  # (B, 1, 1)
    return (1 - t) * x_0 + t * x_1


def velocity_target(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """Velocity target: v = x_1 - x_0 (noise - data)."""
    return x_1 - x_0


def euler_step(x_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
    """Single Euler step: x_{t+dt} = x_t + dt * v_t."""
    return x_t + dt * v_t


@torch.no_grad()
def euler_solve(
    model_fn,
    x_T: torch.Tensor,
    steps: int = 50,
) -> torch.Tensor:
    """Euler ODE solve from t=1 (noise) to t=0 (data).

    Args:
        model_fn: Callable(x_t, t) -> v_t. Should handle CFG internally.
        x_T: (B, N, D) initial noise at t=1
        steps: Number of Euler steps

    Returns:
        (B, N, D) denoised data at t=0
    """
    sigmas = torch.linspace(1, 0, steps + 1, device=x_T.device)
    x_t = x_T

    for i in range(steps):
        t = sigmas[i].expand(x_T.shape[0])
        dt = sigmas[i + 1] - sigmas[i]  # negative
        v_t = model_fn(x_t, t)
        x_t = x_t + dt * v_t

    return x_t


def estimate_x0(
    x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor,
) -> torch.Tensor:
    """Estimate x_0 from current noisy sample and predicted velocity.

    From x_t = (1-t)*x_0 + t*x_1 and v = x_1 - x_0:
        x_0 = x_t - t * v_pred

    Used for the geometry regularizer (single-step denoise approximation).
    """
    t = t[:, None, None]
    return x_t - t * v_pred
