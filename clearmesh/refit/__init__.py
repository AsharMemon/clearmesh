"""ClearMesh primitive-refit module.

Stages (compose or use independently):

  cad_recode         point cloud -> CadQuery script       (CC-BY-NC)
  light_sq           mesh -> superquadric decomposition   (port of paper)
  watertight_select  candidate faces -> watertight subset via BLP
                     (port of Liu 2023 §3.3 Selection Module)
"""

from clearmesh.refit.cad_recode import CadRecodeRefiner, CadRecodeResult
from clearmesh.refit.light_sq import LightSQRefiner, SuperQuadric, LightSQResult
from clearmesh.refit.triangle_split import split_candidates
from clearmesh.refit.watertight_select import select_watertight, SelectionResult

__all__ = [
    "CadRecodeRefiner", "CadRecodeResult",
    "LightSQRefiner", "SuperQuadric", "LightSQResult",
    "split_candidates",
    "select_watertight", "SelectionResult",
]
