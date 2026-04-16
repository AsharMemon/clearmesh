"""X-Part part decomposition: single coarse mesh → multiple semantic parts.

X-Part (Tencent Hunyuan3D-Part, arxiv:2509.08643) is a controllable generative
model that decomposes a holistic 3D mesh into semantically meaningful and
structurally coherent parts at high geometric fidelity. Unlike PartCrafter
(which takes an image), X-Part takes an *existing* mesh — natural fit for our
pipeline since TRELLIS.2 has already produced a coarse mesh at this point.

Architecture:
  - P3-SAM: 3D point-level segmentation → per-part bounding boxes
  - X-Part (PartFormer): bbox-conditioned part generation
  - Both are bundled inside `PartFormerPipeline.from_pretrained("tencent/Hunyuan3D-Part")`
    so demo.py only exposes a single entry point.

Reference inference command (from XPart/README.md, upstream):
    python demo.py \\
        --config partgen/config/infer.yaml \\
        --mesh_path input.glb \\
        --save_dir results/

Outputs (per the demo.py we inspected upstream):
    train_cfg_{cfg:04f}_*_boxgpt_{uid}.glb
    - processed object mesh (decomposed)
    - output bounding boxes
    - GT bounding boxes
    - exploded object visualization

For our wrapper we only consume the processed object mesh.

Hardware: not documented upstream; budget ~24 GB VRAM similar to PartCrafter.

Status: SCAFFOLD. Wrapper structure validated offline against the published
demo.py CLI contract, but inference has not yet been run end-to-end against a
TRELLIS.2 coarse mesh — needs a GPU session with:
  - Hunyuan3D-Part repo cloned to `xpart_dir`
  - Sonata deps installed (see upstream README)
  - `tencent/Hunyuan3D-Part` checkpoint downloaded

Usage:
    from clearmesh.xpart import XPartDecomposer

    decomposer = XPartDecomposer(xpart_dir="/workspace/Hunyuan3D-Part/XPart")
    parts = decomposer.decompose(coarse_mesh)  # trimesh or GLB path
    for part in parts:
        print(f"{part.label}: {part.mesh.vertices.shape[0]} verts")
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import trimesh
from PIL import Image

# Share the MeshPart dataclass + hard/organic classifier with the PartCrafter path.
# Both decomposers produce the same output contract so downstream stages
# (super-res, retopo, repair) don't care which produced the parts.
from clearmesh.partcrafter.decompose import MeshPart, PartDecomposer


class XPartDecomposer:
    """Part decomposition via X-Part (Hunyuan3D-Part).

    Subprocess-based wrapper around `XPart/demo.py` — same pattern as the
    PartCrafter wrapper. Isolates upstream deps (Sonata, Hunyuan3D-2.1) from
    the main ClearMesh env, at the cost of one extra Python start per call.

    Args:
        xpart_dir: Path to cloned Hunyuan3D-Part/XPart directory.
        config_path: Config YAML (relative to xpart_dir or absolute).
            Defaults to "partgen/config/infer.yaml" as shipped upstream.
        checkpoint: Optional explicit checkpoint path to override config.
        device: CUDA device (demo.py always uses cuda).
    """

    def __init__(
        self,
        xpart_dir: str = "/workspace/Hunyuan3D-Part/XPart",
        config_path: str = "partgen/config/infer.yaml",
        checkpoint: str | None = None,
        device: str = "cuda",
    ):
        self.xpart_dir = xpart_dir
        self.config_path = config_path
        self.checkpoint = checkpoint
        self.device = device

    def is_available(self) -> bool:
        """Check if X-Part is installed and the demo entry point is present."""
        return (
            os.path.isdir(self.xpart_dir)
            and os.path.isfile(os.path.join(self.xpart_dir, "demo.py"))
        )

    def decompose(
        self,
        mesh: trimesh.Trimesh | str | Path,
        image: Image.Image | str | None = None,  # unused; signature-compat with PartDecomposer
        **_unused,  # num_parts, min_parts, etc. — not supported by X-Part CLI
    ) -> list[MeshPart]:
        """Decompose a coarse mesh into semantic 3D parts.

        Args:
            mesh: Coarse mesh (trimesh object or path to GLB/OBJ/etc).
                X-Part takes a GLB; trimesh inputs are written to a temp file.
            image: Kept for signature compatibility with PartDecomposer —
                X-Part does not use an image reference.
            **_unused: Accepts (and ignores) PartCrafter kwargs like
                `num_parts`, so `pipeline.py` can call either decomposer with
                the same args.

        Returns:
            List of MeshPart objects, one per semantic part.
        """
        if not self.is_available():
            raise RuntimeError(
                f"X-Part not found at {self.xpart_dir}. "
                "Clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part "
                "and install deps per XPart/README.md."
            )

        # Resolve mesh input to a file path demo.py can read.
        if isinstance(mesh, trimesh.Trimesh):
            tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
            tmp.close()
            mesh.export(tmp.name)
            mesh_path = tmp.name
            cleanup_input = True
        else:
            mesh_path = str(mesh)
            cleanup_input = False

        try:
            with tempfile.TemporaryDirectory() as save_dir:
                cmd = [
                    sys.executable,
                    "demo.py",
                    "--config", self.config_path,
                    "--mesh_path", mesh_path,
                    "--save_dir", save_dir,
                ]
                if self.checkpoint is not None:
                    cmd.extend(["--ckpt", self.checkpoint])

                try:
                    subprocess.run(
                        cmd,
                        cwd=self.xpart_dir,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=900,  # 15 min ceiling per call
                    )
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        f"X-Part demo.py failed (exit {e.returncode}).\n"
                        f"stdout tail: {e.stdout[-500:] if e.stdout else ''}\n"
                        f"stderr tail: {e.stderr[-500:] if e.stderr else ''}"
                    ) from e

                parts = self._load_parts(save_dir)
        finally:
            if cleanup_input:
                try:
                    os.unlink(mesh_path)
                except OSError:
                    pass

        return parts

    def _load_parts(self, save_dir: str) -> list[MeshPart]:
        """Parse demo.py's output directory into a list of MeshPart.

        Upstream writes files matching `train_cfg_*_boxgpt_*.glb` plus bbox
        auxiliaries. We only consume the decomposed object mesh; bboxes /
        exploded visualizations are skipped.

        The single output GLB may contain multiple named sub-meshes (one per
        semantic part) packed as a trimesh.Scene. We flatten that into a list.
        """
        save_path = Path(save_dir)

        # Find the *object* GLB. Upstream names include "boxgpt" for bboxes
        # and "explode" for the exploded visualization — skip those.
        candidates = []
        for glb in save_path.rglob("*.glb"):
            name = glb.name.lower()
            if "box" in name or "bbox" in name or "explode" in name:
                continue
            candidates.append(glb)

        if not candidates:
            raise RuntimeError(
                f"X-Part produced no part GLBs in {save_dir}. "
                f"Contents: {sorted(p.name for p in save_path.rglob('*'))}"
            )

        # Prefer the most recent (highest mtime) if multiple survive the filter.
        candidates.sort(key=lambda p: p.stat().st_mtime)
        parts_glb = candidates[-1]

        loaded = trimesh.load(str(parts_glb), force=None)

        parts: list[MeshPart] = []
        if isinstance(loaded, trimesh.Scene):
            # Multi-part scene — one MeshPart per named geometry.
            for i, (name, geom) in enumerate(loaded.geometry.items()):
                if not isinstance(geom, trimesh.Trimesh):
                    continue
                label = _clean_label(name, i)
                parts.append(MeshPart(mesh=geom, label=label, part_id=i))
        elif isinstance(loaded, trimesh.Trimesh):
            # Single concatenated mesh — upstream also outputs this variant.
            # Treat it as a single-part decomposition; the caller can decide
            # whether to re-run with a finer config.
            parts.append(MeshPart(mesh=loaded, label=parts_glb.stem, part_id=0))
        else:
            raise RuntimeError(
                f"X-Part output {parts_glb} produced unexpected type {type(loaded)}"
            )

        # Share the PartCrafter classifier so part categories are consistent
        # regardless of which decomposer produced them.
        for part in parts:
            part.is_hard_surface = PartDecomposer._classify_surface(part)

        return parts

    def decompose_or_passthrough(
        self,
        image: Image.Image | str,
        mesh: trimesh.Trimesh,
        **kwargs,
    ) -> list[MeshPart]:
        """Decompose if X-Part is available, otherwise wrap mesh as single part.

        Signature matches PartDecomposer.decompose_or_passthrough so pipeline.py
        can call either backend interchangeably.
        """
        if self.is_available():
            try:
                return self.decompose(mesh, image=image, **kwargs)
            except Exception as e:
                print(f"X-Part failed: {e}. Using single-part fallback.")

        return [MeshPart(mesh=mesh, label="whole", part_id=0)]


def _clean_label(raw: str, idx: int) -> str:
    """Normalize a scene-geometry name into a short semantic label.

    Upstream doesn't guarantee clean names — may include file prefixes,
    numeric IDs, or hashes. Strip common noise; fall back to `part_{idx}`.
    """
    s = raw.strip().lower()
    for noise in ("train_cfg_", "_boxgpt", "mesh_", "geometry_"):
        s = s.replace(noise, "")
    s = s.strip("_0123456789-").strip("_")
    return s or f"part_{idx}"
