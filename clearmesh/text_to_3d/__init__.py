"""ClearMesh Text-to-3D Module.

Two-step text-to-3D generation:
  Text prompt → FLUX.1-schnell (4 steps) → Image → ClearMesh Pipeline → 3D model

Also supports text-guided editing:
  Source mesh + text instruction → InstructPix2Pix → edit image → Easy3E → edited mesh

Usage:
    from clearmesh.text_to_3d import TextTo3D

    generator = TextTo3D()
    result = generator.generate("a fierce red dragon", output_path="dragon.glb")
"""

from clearmesh.text_to_3d.generate import TextTo3D

__all__ = ["TextTo3D"]
