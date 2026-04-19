"""DualPrim scene I/O: JSON primitives + trajectory snapshots.

Why JSON instead of .pt:
  - downstream training pipelines need a portable, language-agnostic
    format (Python dataset loader, potentially JS visualizer)
  - .pt files are fine for optimizer-resume mid-training, but training
    a feedforward model across many scenes wants something that can be
    inspected, filtered, and sharded

The JSON schema is intentionally mirror-of-DualPrimitive: one record
per live primitive with field names matching the dataclass. That keeps
the loader trivial.

Trajectory snapshots are just primitives.json with an "iteration" field
and a standard name ("step_{N:06d}.json"). The dataset consumer stacks
them as (mesh_id, seed, step) tuples.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from clearmesh.dualprim.params import DualPrimConfig
from clearmesh.dualprim.types import (
    DualPrimitive,
    DualPrimScene,
    DUAL_PRIM_DIM,
)


# ---------------------------------------------------------------------
# Canonical trajectory checkpoints
# ---------------------------------------------------------------------
# Friend's guidance: we want intermediate states, not just endpoints.
# Five log-spaced checkpoints over a 15k-iter run captures enough
# of the optimization trajectory to train a warm-start predictor on
# "any point along the refinement curve" rather than "only the final
# converged point".
DEFAULT_TRAJECTORY_FRACTIONS = (1/15, 3/15, 6/15, 10/15, 1.0)


def canonical_trajectory_iters(num_iterations: int) -> list[int]:
    """Canonical snapshot iters: ~log-spaced across training.

    For 15000 iters → [1000, 3000, 6000, 10000, 15000]
    For 5000 iters  → [333, 1000, 2000, 3333, 5000] (rounded)

    The last iter is always included so callers don't need to special-case
    the final state.
    """
    iters = sorted({max(1, int(round(f * num_iterations)))
                    for f in DEFAULT_TRAJECTORY_FRACTIONS})
    # Always include the terminal iter, never iter 0 (useless).
    if num_iterations not in iters:
        iters.append(num_iterations)
    return sorted(i for i in iters if 0 < i <= num_iterations)


# ---------------------------------------------------------------------
# Scene → JSON
# ---------------------------------------------------------------------

def _tensor_to_list(x):
    """Detach + CPU + to-python-list. Scalars become a 0-d list."""
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return x  # scalar-ish


def primitives_to_dicts(scene: DualPrimScene) -> list[dict]:
    """Return each live primitive as a plain-dict (JSON-ready)."""
    out = []
    for dp in scene.live_primitives():
        d = {}
        for k, v in asdict(dp).items():
            d[k] = _tensor_to_list(v)
        out.append(d)
    return out


def save_scene_json(
    scene: DualPrimScene,
    path: str | Path,
    *,
    iteration: Optional[int] = None,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Write the scene's live primitives to a JSON file.

    Schema:
      {
        "iteration": int | null,
        "num_alive": int,
        "num_slots": int,
        "primitives": [ { "psq_scale": [...], ... }, ... ],
        "metadata": {...}
      }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iteration,
        "num_alive": int(scene.alive.sum().item()),
        "num_slots": int(scene.K),
        "primitives": primitives_to_dicts(scene),
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ---------------------------------------------------------------------
# JSON → Scene
# ---------------------------------------------------------------------

_REQUIRED_PRIM_KEYS = [
    "psq_scale", "nsq_scale",
    "psq_shape", "nsq_shape",
    "alpha", "theta",
    "psq_translation", "nsq_translation",
    "psq_rotation_rad", "nsq_rotation_rad",
    "color",
]


def _prim_dict_to_row(d: dict, device) -> torch.Tensor:
    """Pack one primitive dict into a (DUAL_PRIM_DIM,) tensor row.

    Raises on missing keys so bad data fails loudly rather than
    silently zeroing fields.
    """
    missing = [k for k in _REQUIRED_PRIM_KEYS if k not in d]
    if missing:
        raise ValueError(f"primitive dict missing required keys: {missing}")
    dp = DualPrimitive(
        psq_scale=torch.tensor(d["psq_scale"], dtype=torch.float32, device=device),
        nsq_scale=torch.tensor(d["nsq_scale"], dtype=torch.float32, device=device),
        psq_shape=torch.tensor(d["psq_shape"], dtype=torch.float32, device=device),
        nsq_shape=torch.tensor(d["nsq_shape"], dtype=torch.float32, device=device),
        alpha=torch.tensor(d["alpha"], dtype=torch.float32, device=device),
        theta=torch.tensor(d["theta"], dtype=torch.float32, device=device),
        psq_translation=torch.tensor(d["psq_translation"], dtype=torch.float32, device=device),
        nsq_translation=torch.tensor(d["nsq_translation"], dtype=torch.float32, device=device),
        psq_rotation_rad=torch.tensor(d["psq_rotation_rad"], dtype=torch.float32, device=device),
        nsq_rotation_rad=torch.tensor(d["nsq_rotation_rad"], dtype=torch.float32, device=device),
        color=torch.tensor(d["color"], dtype=torch.float32, device=device),
    )
    return dp.to_vector()


def load_scene_from_json(
    path: str | Path,
    config: DualPrimConfig,
    *,
    device: str = "cuda",
    pad_to_K: Optional[int] = None,
) -> DualPrimScene:
    """Load a DualPrimScene from a primitives JSON (final or trajectory).

    Args:
      path: path to a JSON produced by save_scene_json or run_canary.
      config: DualPrimConfig — used to allocate the lighting MLP. Other
        config fields (seed, init ranges, etc.) are irrelevant here.
      device: torch device.
      pad_to_K: if set, pad the scene out to this many slots by
        appending inactive (alive=False) zero rows. Useful when the
        refiner expects a fixed K but the loaded trajectory has fewer
        live primitives after pruning. Default: pack to live count.

    Returns a scene with:
      - params.requires_grad = True
      - alive[:N_live] = True, alive[N_live:] = False (if padded)
      - lighting_mlp freshly initialized (NOT loaded — that's a separate
        concern handled by the .pt checkpoint path)
    """
    path = Path(path)
    with open(path) as f:
        payload = json.load(f)

    prims = payload.get("primitives", [])
    n_live = len(prims)
    if n_live == 0:
        raise ValueError(f"{path}: no live primitives in JSON")

    k = max(n_live, pad_to_K or 0)
    params = torch.zeros((k, DUAL_PRIM_DIM), dtype=torch.float32, device=device)
    for i, d in enumerate(prims):
        params[i] = _prim_dict_to_row(d, device=device)
    params = params.clone().requires_grad_(True)

    # Build the lighting MLP fresh. The trajectory snapshot does NOT
    # serialize MLP weights — if Gate 2 needs them, we'll load the
    # sibling .pt checkpoint. For warm-start-only use cases (where the
    # downstream refiner retrains lighting from scratch), that's fine.
    lighting_mlp = None
    if config.lighting_mlp_layers > 0 and config.lighting_mlp_hidden > 0:
        from clearmesh.dualprim.renderer import LightingMLP
        lighting_mlp = LightingMLP(
            hidden=config.lighting_mlp_hidden,
            layers=config.lighting_mlp_layers,
        ).to(device)

    scene = DualPrimScene(params=params, lighting_mlp=lighting_mlp)
    scene.alive = torch.zeros(k, dtype=torch.bool, device=device)
    scene.alive[:n_live] = True
    return scene
