"""DualPrim: compact 3D reconstruction with positive + negative primitives.

Port of arXiv 2603.16133 (Meng et al., March 2026). Per-scene
optimization via differentiable volumetric rendering of superquadric
primitive pairs.

Package layout:
  params.py          — DualPrimConfig with all knobs (paper + guesses)
  types.py           — DualPrimitive dataclass + DualPrimScene state
  superquadric.py    — SQ implicit, P_E gate, combined field (Eq 2-6)
  renderer.py        — NeuS-style volumetric renderer (Eq 1, 7-11)
  losses.py          — six loss terms (Eq 12-18)
  optimize_scene.py  — Adam loop + adaptive pruning (§4.2)
  export.py          — Boolean difference mesh export (§3.3)
"""

from clearmesh.dualprim.params import DualPrimConfig
from clearmesh.dualprim.types import DualPrimitive, DualPrimScene
from clearmesh.dualprim.optimize_scene import (
    init_scene, train, prune, clip_to_ranges,
    RaySampleBatch, TrainingState,
)
from clearmesh.dualprim.export import (
    export_dual_primitive, export_scene,
    tessellate_superquadric, boolean_difference,
)

__all__ = [
    "DualPrimConfig",
    "DualPrimitive", "DualPrimScene",
    "init_scene", "train", "prune", "clip_to_ranges",
    "RaySampleBatch", "TrainingState",
    "export_dual_primitive", "export_scene",
    "tessellate_superquadric", "boolean_difference",
]
