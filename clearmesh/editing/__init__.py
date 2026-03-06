"""ClearMesh Editing Module — Easy3E-based 3D editing.

Training-free geometry editing using TRELLIS.2's latent space (SLAT),
with optional Ctrl-Adapter for normal-guided texture generation.

Architecture (from Easy3E, arxiv 2602.21499v1):
  1. SLAT Encoder: mesh → sparse latent (voxel structure + per-voxel features)
  2. Voxel FlowEdit: flow-matching ODE to edit structure (training-free)
  3. SLAT Repainting: preserve unedited regions, regenerate edited (training-free)
  4. Ctrl-Adapter: normal-guided multi-view texture (trainable, ~20 GPU-hrs)
  5. Image Edit: InstructPix2Pix wrapper for text→edit image

Usage:
    from clearmesh.editing import Easy3EEditor

    editor = Easy3EEditor(trellis2_dir="/workspace/TRELLIS.2")
    result = editor.edit(
        source_mesh="model.glb",
        edit_image="edited_view.png",
        edit_mask=None,  # auto-detected
    )
"""

from clearmesh.editing.easy3e import Easy3EEditor

__all__ = ["Easy3EEditor"]
