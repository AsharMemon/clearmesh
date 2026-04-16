"""End-to-end test for text-guided 3D editing: the "add wings" demo.

Takes the Easy3E paper's actual workflow (image → SLAT → edit → SLAT →
mesh) rather than mesh encoding (which TRELLIS.2-4B doesn't support).

Slow: runs full TRELLIS.2 source generation + IP2P inference + edit
flow ODE + decoding. Budget ~3-5 min on H100 NVL 94GB.

Run on pod:

    source /opt/conda/etc/profile.d/conda.sh && conda activate trellis2
    export PYTHONPATH=/workspace/TRELLIS.2:$PYTHONPATH
    cd /workspace/clearmesh
    pytest -q tests/e2e/test_edit_text_guided.py -m "slow and gpu and trellis2" -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = [pytest.mark.slow, pytest.mark.gpu, pytest.mark.trellis2]


def test_add_wings_demo(trellis2_pipeline, tmp_output_dir):
    """The demo: source image → TRELLIS.2 SLAT → edit with 'add wings'
    instruction → edited mesh. Verifies non-degenerate output + writes
    artifacts for visual inspection.
    """
    import numpy as np
    import trimesh
    from PIL import Image
    from clearmesh.editing import Easy3EEditor, EditOptions

    # --- 1. Source image ---
    # Use TRELLIS.2 example asset if available, else synthetic.
    ex_path = Path("/workspace/TRELLIS.2/assets/example_image/T.png")
    if ex_path.exists():
        source_img = Image.open(ex_path).convert("RGB")
    else:
        arr = np.full((512, 512, 3), 128, dtype=np.uint8)
        arr[180:340, 200:312] = (220, 80, 80)
        source_img = Image.fromarray(arr)

    source_path = tmp_output_dir / "source_input.png"
    source_img.save(source_path)

    # --- 2. Instantiate editor with shared pipeline ---
    editor = Easy3EEditor(
        trellis2_dir="/workspace/TRELLIS.2",
        model_dir="/workspace/models/trellis2-4b",
        device="cuda",
        pipeline=trellis2_pipeline,
    )

    options = EditOptions(
        num_flow_steps=12,
        num_repaint_steps=12,
        guidance_scale=7.5,
        text_num_steps=20,
        text_image_guidance=1.5,
        text_guidance_scale=7.5,
        enable_repair=True,
        enable_texture=False,
        export_format="glb",
    )

    edited_path = tmp_output_dir / "edited.glb"
    result = editor.edit_from_source_image(
        source_image=source_img,
        instruction="add large feathered wings",
        output_path=str(edited_path),
        options=options,
    )

    # --- 3. Verify ---
    assert result.mesh is not None, "Editor returned no mesh"
    verts = result.mesh.vertices
    assert verts.shape[0] > 100, f"Edited mesh too small: {verts.shape[0]} verts"
    assert np.all(np.isfinite(verts)), "NaN/inf vertices in edited mesh"
    assert edited_path.exists(), "Edited GLB was not written"

    # Reload and re-verify
    reloaded = trimesh.load(str(edited_path), force="mesh")
    assert reloaded.vertices.shape[0] > 100

    print(f"\n=== ADD WINGS DEMO ARTIFACTS ===")
    print(f"  source image: {source_path}")
    print(f"  edited mesh:  {edited_path}  ({verts.shape[0]} verts, {result.mesh.faces.shape[0]} faces)")
    print(f"  timings: {result.timings}")
