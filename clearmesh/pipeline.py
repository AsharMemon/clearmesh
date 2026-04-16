#!/usr/bin/env python3
"""ClearMesh Pipeline: Image → Print-ready 3D model.

Full pipeline stages:
  1. Background removal (rembg)
  2. Stage 1: Coarse generation (TRELLIS.2 4B, pre-trained)
  3. Part decomposition (PartCrafter, optional — kitbashing mode)
  4. Stage 2: Geometric refinement (UltraShape 1.0, pre-trained — arxiv:2512.21185)
  5. Isosurface extraction (NDC/FlexiCubes — sharp edges, per-part selective)
  6. Geometry super-resolution (SuperCarver / CraftsMan3D, optional)
  7. Retopology (BPT, optional — for digital/game-ready output)
  8. Mesh repair + print-readiness (PyMeshFix, orientation, drain holes)
  9. Scale normalization (28mm, 32mm, 54mm miniature scales)
  10. PBR textures (optional — for digital variants)
  11. Auto-rigging (Puppeteer/UniRig, optional)
  12. Export (STL, GLB, FBX)

Usage:
    from clearmesh.pipeline import ClearMeshPipeline

    pipeline = ClearMeshPipeline(
        ultrashape_dir='/workspace/UltraShape-1.0',
        ultrashape_checkpoint='/workspace/checkpoints/ultrashape_v1.pt',
    )

    result = pipeline.generate('photo.png', options={
        'resolution': 512,
        'enable_refinement': True,
        'target_scale': '32mm',
        'export_format': 'stl',
    })

CLI:
    python -m clearmesh.pipeline --input photo.png --output model.stl --scale 32mm
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from clearmesh.mesh.export import export_mesh
from clearmesh.mesh.repair import repair_mesh, validate_for_printing, orient_for_printing, full_print_preparation
from clearmesh.utils.background_removal import remove_background
from clearmesh.utils.scale import scale_to_preset


@dataclass
class GenerationOptions:
    """Options for the generation pipeline."""

    # Stage 1: Coarse generation
    resolution: int = 512  # 512 | 1024 | 1536

    # Part decomposition (PartCrafter)
    enable_part_decomposition: bool = False  # Kitbashing mode
    num_parts: int | None = None  # Auto-detect if None

    # Stage 2: Refinement (UltraShape)
    enable_refinement: bool = True
    refinement_steps: int = 50  # 50=quality, 25=fast
    refinement_octree_res: int = 512  # 512 (fast) | 1024 (max detail)
    refinement_seed: int = 42
    refinement_low_vram: bool = False  # Enable for GPUs with <24GB VRAM

    # Geometry super-resolution (SuperCarver / CraftsMan3D)
    enable_super_resolution: bool = False
    super_resolution_detail: str = "medium"  # low | medium | high

    # Retopology (BPT)
    enable_retopology: bool = False  # Only for digital/game-ready
    retopo_target_faces: int = 8000

    # Mesh repair + print prep
    orient_for_print: bool = True
    drain_holes: bool = False
    drain_hole_radius_mm: float = 1.5

    # Scale
    target_scale: str | None = None  # 28mm | 32mm | 54mm | 75mm | None
    add_base: bool = False
    hollow: bool = False
    wall_thickness_mm: float = 1.5

    # PBR textures
    enable_textures: bool = False  # For digital variants

    # Auto-rigging (optional)
    enable_rigging: bool = False
    rigging_method: str = "puppeteer"  # puppeteer | unirig | humanrig

    # Export
    export_format: str = "glb"  # stl | glb | obj | fbx

    # Background removal
    skip_background_removal: bool = False


@dataclass
class GenerationResult:
    """Result from the pipeline."""

    output_path: str
    mesh: trimesh.Trimesh
    parts: list | None = None  # List of MeshPart if decomposed
    skeleton: dict | None = None
    skin_weights: np.ndarray | None = None
    print_report: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


class ClearMeshPipeline:
    """Full ClearMesh generation pipeline.

    Modular design: each component can be enabled/disabled independently.
    All optional stages default to off and are only loaded when requested.
    """

    def __init__(
        self,
        ultrashape_dir: str = "/workspace/UltraShape-1.0",
        ultrashape_checkpoint: str = "/workspace/checkpoints/ultrashape_v1.pt",
        model_dir: str = "/mnt/data",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self._ultrashape_dir = ultrashape_dir
        self._ultrashape_checkpoint = ultrashape_checkpoint

        # All models are lazy-loaded
        self._stage1 = None
        self._stage2 = None
        self._part_decomposer = None
        self._super_resolver = None
        self._retopologizer = None
        self._rigger = None

    # --- Lazy-loaded model properties ---

    @property
    def stage1(self):
        """Lazy-load TRELLIS.2 pipeline."""
        if self._stage1 is None:
            print("Loading TRELLIS.2-4B...")
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            self._stage1 = Trellis2ImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS.2-4B"
            )
            self._stage1.to(self.device)
            print("TRELLIS.2-4B loaded.")
        return self._stage1

    @property
    def stage2(self):
        """Lazy-load Stage 2 refiner (UltraShape 1.0)."""
        if self._stage2 is None:
            from clearmesh.stage2.ultrashape_refiner import UltraShapeRefiner

            self._stage2 = UltraShapeRefiner(
                ultrashape_dir=self._ultrashape_dir,
                checkpoint=self._ultrashape_checkpoint,
                device=self.device,
            )
        return self._stage2

    @property
    def part_decomposer(self):
        """Lazy-load PartCrafter."""
        if self._part_decomposer is None:
            from clearmesh.partcrafter.decompose import PartDecomposer
            self._part_decomposer = PartDecomposer(
                partcrafter_dir=os.path.join(self.model_dir, "PartCrafter")
            )
        return self._part_decomposer

    @property
    def super_resolver(self):
        """Lazy-load geometry super-resolution."""
        if self._super_resolver is None:
            from clearmesh.supercarver.super_resolve import GeometrySuperResolver
            self._super_resolver = GeometrySuperResolver(model_dir=self.model_dir)
        return self._super_resolver

    @property
    def retopologizer(self):
        """Lazy-load retopology model."""
        if self._retopologizer is None:
            from clearmesh.retopology.retopo import Retopologizer
            self._retopologizer = Retopologizer(model_dir=self.model_dir)
        return self._retopologizer

    def _get_rigger(self, method: str):
        """Get auto-rigger with specified method."""
        if self._rigger is None:
            print(f"Loading auto-rigger ({method})...")
            from clearmesh.rigging.auto_rigger import AutoRigger
            self._rigger = AutoRigger(method=method, model_dir=self.model_dir)
            print("Auto-rigger loaded.")
        return self._rigger

    # --- Main pipeline ---

    def generate(
        self,
        image_path: str,
        output_path: str | None = None,
        options: GenerationOptions | dict | None = None,
    ) -> GenerationResult:
        """Full pipeline: image → print-ready 3D model.

        Args:
            image_path: Path to input image
            output_path: Output file path (auto-generated if None)
            options: Generation options (dict or GenerationOptions)

        Returns:
            GenerationResult with mesh, paths, and metadata
        """
        if isinstance(options, dict):
            options = GenerationOptions(**options)
        elif options is None:
            options = GenerationOptions()

        timings = {}

        if output_path is None:
            stem = Path(image_path).stem
            output_path = f"{stem}_clearmesh.{options.export_format}"

        # === 1. Background removal ===
        t0 = time.time()
        image = Image.open(image_path)
        if not options.skip_background_removal:
            image = remove_background(image)
        timings["background_removal"] = time.time() - t0

        # === 2. Stage 1 — Coarse generation (TRELLIS.2) ===
        t0 = time.time()
        with torch.no_grad():
            coarse_result = self.stage1.run(image, resolution=options.resolution)
        coarse_mesh_raw = coarse_result[0]
        timings["stage1_coarse"] = time.time() - t0

        from clearmesh.mesh.extraction import extract_from_ovoxel
        mesh = extract_from_ovoxel(coarse_mesh_raw)

        # Extract PBR textures if available
        pbr_textures = None
        if options.enable_textures:
            from clearmesh.texture.pbr import PBRTextures
            pbr_textures = PBRTextures.from_pipeline_output(coarse_mesh_raw)

        # === 3. Part decomposition (PartCrafter, optional) ===
        parts = None
        if options.enable_part_decomposition:
            t0 = time.time()
            parts = self.part_decomposer.decompose_or_passthrough(
                image, mesh, num_parts=options.num_parts
            )
            timings["part_decomposition"] = time.time() - t0
            print(f"Decomposed into {len(parts)} parts: {[p.label for p in parts]}")

        # === 4. Stage 2 — Geometric refinement (UltraShape) ===
        if options.enable_refinement:
            t0 = time.time()
            if parts:
                for part in parts:
                    part.mesh = self._run_stage2(part.mesh, image, options)
            else:
                mesh = self._run_stage2(mesh, image, options)
            timings["stage2_refinement"] = time.time() - t0

        # === 5. Isosurface extraction (NDC/FlexiCubes — per-part selective) ===
        # UltraShape output is already high-detail marching cubes; re-voxelizing
        # at a lower resolution would destroy detail. Only run edge sharpening
        # when refinement is disabled (coarse mesh needs help) or explicitly
        # requested for hard-surface parts.
        if not options.enable_refinement:
            t0 = time.time()
            if parts:
                for part in parts:
                    if part.category == "hard":
                        part.mesh = self._sharpen_edges(part.mesh)
            else:
                mesh = self._sharpen_edges(mesh)
            timings["edge_sharpening"] = time.time() - t0

        # === 6. Geometry super-resolution (optional) ===
        if options.enable_super_resolution and self.super_resolver.is_available():
            t0 = time.time()
            if parts:
                for part in parts:
                    if self.super_resolver.should_apply(part.label):
                        part.mesh = self.super_resolver.super_resolve(
                            part.mesh, detail_level=options.super_resolution_detail
                        )
            else:
                mesh = self.super_resolver.super_resolve(
                    mesh, detail_level=options.super_resolution_detail
                )
            timings["super_resolution"] = time.time() - t0

        # === 7. Retopology (optional, for digital/game-ready) ===
        if options.enable_retopology and self.retopologizer.is_available():
            t0 = time.time()
            if parts:
                for part in parts:
                    part.mesh = self.retopologizer.retopologize(
                        part.mesh, target_faces=options.retopo_target_faces
                    )
            else:
                mesh = self.retopologizer.retopologize(
                    mesh, target_faces=options.retopo_target_faces
                )
            timings["retopology"] = time.time() - t0

        # Reassemble parts into single mesh if decomposed
        if parts:
            mesh = trimesh.util.concatenate([p.mesh for p in parts])

        # === 8. Mesh repair + print preparation ===
        t0 = time.time()
        mesh = full_print_preparation(
            mesh,
            orient=options.orient_for_print,
            verbose=True,
        )
        timings["mesh_repair"] = time.time() - t0

        # === 9. Scale normalization ===
        if options.target_scale:
            t0 = time.time()
            mesh = scale_to_preset(mesh, options.target_scale)

            if options.add_base:
                from clearmesh.utils.scale import add_base
                mesh = add_base(mesh)

            if options.hollow:
                from clearmesh.utils.scale import hollow_mesh
                mesh = hollow_mesh(mesh, wall_thickness_mm=options.wall_thickness_mm)

                if options.drain_holes:
                    from clearmesh.mesh.repair import add_drain_holes
                    mesh = add_drain_holes(mesh, hole_radius_mm=options.drain_hole_radius_mm)

            timings["scaling"] = time.time() - t0

        # === 10. PBR textures (optional) ===
        if options.enable_textures and pbr_textures is not None and pbr_textures.has_textures:
            mesh = pbr_textures.apply(mesh)

        # === 11. Auto-rigging (optional) ===
        skeleton = None
        skin_weights = None
        if options.enable_rigging:
            t0 = time.time()
            rigger = self._get_rigger(options.rigging_method)
            skeleton, skin_weights = rigger.rig(mesh)
            timings["rigging"] = time.time() - t0

        # === 12. Export ===
        t0 = time.time()
        texture = pbr_textures.albedo if (pbr_textures and pbr_textures.has_textures) else None
        exported_path = export_mesh(
            mesh,
            output_path,
            format=options.export_format,
            texture=texture,
            skeleton=skeleton,
            skin_weights=skin_weights,
        )
        timings["export"] = time.time() - t0

        print_report = validate_for_printing(mesh)
        timings["total"] = sum(timings.values())

        print(f"\nGeneration complete in {timings['total']:.1f}s")
        print(f"Output: {exported_path}")
        print(f"Mesh: {mesh.vertices.shape[0]:,} verts, {mesh.faces.shape[0]:,} faces")
        print(f"Watertight: {mesh.is_watertight}")
        if print_report["issues"]:
            print(f"Print issues: {print_report['issues']}")

        return GenerationResult(
            output_path=exported_path,
            mesh=mesh,
            parts=parts,
            skeleton=skeleton,
            skin_weights=skin_weights,
            print_report=print_report,
            timings=timings,
        )

    # --- Internal helpers ---

    def _run_stage2(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        options: GenerationOptions,
    ) -> trimesh.Trimesh:
        """Run Stage 2 refinement on a mesh using UltraShape.

        Args:
            mesh: Coarse mesh from TRELLIS.2.
            image: Reference image (RGBA, background already removed).
            options: Generation options.

        Returns:
            Refined trimesh with significantly more detail (~14% more verts typical).
        """
        # Ensure image is RGBA (background should already be removed upstream)
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # UltraShape's refiner handles its own VRAM mode
        if options.refinement_low_vram and not self.stage2.low_vram:
            self.stage2.low_vram = True

        return self.stage2.refine(
            coarse_mesh=mesh,
            reference_image=image,
            num_steps=options.refinement_steps,
            octree_resolution=options.refinement_octree_res,
            seed=options.refinement_seed,
        )

    def _sharpen_edges(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Post-process with NDC for sharper edges (if available)."""
        try:
            from clearmesh.mesh.extraction import extract_ndc
            R = 256
            voxel_grid = mesh.voxelized(pitch=2.0 / R)
            sdf = voxel_grid.matrix.astype(np.float32)
            return extract_ndc(sdf)
        except Exception:
            return mesh


def main():
    parser = argparse.ArgumentParser(description="ClearMesh: Image → 3D Model")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--format", type=str, default="glb", choices=["stl", "glb", "obj", "fbx"])
    parser.add_argument("--resolution", type=int, default=512, choices=[512, 1024, 1536])
    parser.add_argument("--scale", type=str, default=None, choices=["28mm", "32mm", "54mm", "75mm"])
    parser.add_argument(
        "--ultrashape-dir",
        type=str,
        default="/workspace/UltraShape-1.0",
        help="Path to cloned UltraShape-1.0 repo",
    )
    parser.add_argument(
        "--ultrashape-checkpoint",
        type=str,
        default="/workspace/checkpoints/ultrashape_v1.pt",
        help="Path to ultrashape_v1.pt checkpoint",
    )
    parser.add_argument("--no-refinement", action="store_true")
    parser.add_argument("--octree-res", type=int, default=512, choices=[512, 1024])
    parser.add_argument("--low-vram", action="store_true", help="Enable CPU offload for Stage 2")

    # Optional pipeline stages
    parser.add_argument("--decompose", action="store_true", help="Enable PartCrafter decomposition")
    parser.add_argument("--num-parts", type=int, default=None, help="Part count hint for PartCrafter")
    parser.add_argument("--super-res", action="store_true", help="Enable geometry super-resolution")
    parser.add_argument("--super-res-detail", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--retopo", action="store_true", help="Enable BPT retopology (digital/game-ready)")
    parser.add_argument("--retopo-faces", type=int, default=8000, help="Target face count for retopology")
    parser.add_argument("--textures", action="store_true", help="Enable PBR textures (digital)")
    parser.add_argument("--drain-holes", action="store_true", help="Add drain holes (resin printing)")
    parser.add_argument("--rig", action="store_true", help="Enable auto-rigging (optional)")
    parser.add_argument("--rig-method", type=str, default="puppeteer")
    parser.add_argument("--add-base", action="store_true")
    parser.add_argument("--hollow", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Fast mode (12 diffusion steps)")
    args = parser.parse_args()

    pipeline = ClearMeshPipeline(
        ultrashape_dir=args.ultrashape_dir,
        ultrashape_checkpoint=args.ultrashape_checkpoint,
    )

    options = GenerationOptions(
        resolution=args.resolution,
        enable_part_decomposition=args.decompose,
        num_parts=args.num_parts,
        enable_refinement=not args.no_refinement,
        refinement_steps=25 if args.fast else 50,
        refinement_octree_res=args.octree_res,
        refinement_low_vram=args.low_vram,
        enable_super_resolution=args.super_res,
        super_resolution_detail=args.super_res_detail,
        enable_retopology=args.retopo,
        retopo_target_faces=args.retopo_faces,
        enable_textures=args.textures,
        target_scale=args.scale,
        export_format=args.format,
        add_base=args.add_base,
        hollow=args.hollow,
        drain_holes=args.drain_holes,
        enable_rigging=args.rig,
        rigging_method=args.rig_method,
    )

    result = pipeline.generate(args.input, args.output, options)

    print(f"\n{'='*50}")
    print(f"  ClearMesh Generation Complete")
    print(f"{'='*50}")
    print(f"  Output:     {result.output_path}")
    print(f"  Vertices:   {result.mesh.vertices.shape[0]:,}")
    print(f"  Faces:      {result.mesh.faces.shape[0]:,}")
    print(f"  Watertight: {result.print_report.get('watertight', 'N/A')}")
    print(f"  Printable:  {result.print_report.get('printable', 'N/A')}")
    if result.parts:
        print(f"  Parts:      {len(result.parts)} ({', '.join(p.label for p in result.parts)})")
    if result.skeleton:
        print(f"  Rigged:     Yes ({len(result.skeleton['joints'])} joints)")
    for stage, duration in result.timings.items():
        print(f"  {stage}: {duration:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
