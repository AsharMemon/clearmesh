"""Stage 2: Geometric refinement.

Current production refiner: UltraShape 1.0 (PKU-YuanGroup, arxiv:2512.21185).
Pre-trained voxel-conditioned DiT; no custom training required.

Legacy: RefinementDiT (model.py, model_v2.py) — abandoned April 2026
after UltraShape validation showed comparable quality with zero training cost.
Kept for reference but not used by the main pipeline.
"""

from clearmesh.stage2.ultrashape_refiner import UltraShapeRefiner

__all__ = ["UltraShapeRefiner"]
