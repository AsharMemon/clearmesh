#!/usr/bin/env python3
"""Easy3E demo: image → base mesh → text-guided edit → edited mesh.

End-to-end demo of the Easy3E training-free editing stack on a real GPU.
Takes a single input image, runs it through the full ClearMesh pipeline
(TRELLIS.2 coarse + UltraShape refinement) to produce a base mesh, then
runs ``Easy3EEditor.edit_from_text`` to produce an edited version under a
natural-language instruction.

Artifacts written to ``--output-dir``:

    base_mesh.glb            — ClearMeshPipeline(image → mesh) output
    source_render.png        — front-view render of base_mesh (editor input)
    target_image.png         — InstructPix2Pix(source_render, instruction)
    edited_mesh.glb          — Easy3EEditor.edit output
    report.json              — env + per-stage status/timings + sha256s

The report mirrors ``scripts/e2e_smoke.py`` so downstream tooling can
treat both reports uniformly.

Expected runtime on A100 80GB: ~4–7 min total.
    Stage 1 (base mesh):              90–180s
    Stage 2 (InstructPix2Pix render):  20–40s (first call cold-loads ~4GB)
    Stage 3 (Easy3E edit):            60–120s

Why it's split into explicit stages rather than a single
``editor.edit_from_text(mesh, "…")`` call:
  - We want to save the InstructPix2Pix output for visual inspection —
    if the edit looks wrong, you need to know whether it was the 2D edit
    or the 3D transfer that failed.
  - Base mesh generation and editing run in separate sub-phases with
    different failure modes; separate report entries make triage easy.

Usage:
    # Default: test_mug.png, paint it red
    python scripts/easy3e_demo.py

    # Custom image + instruction
    python scripts/easy3e_demo.py \\
        --input path/to/image.png \\
        --instruction "make the surface look like polished wood" \\
        --view front \\
        --output-dir /tmp/clearmesh_easy3e_demo
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Return hex sha256 of a file; 'missing' if not found."""
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _file_info(path: str) -> dict:
    """Size + sha256 + existence for a produced artifact."""
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    return {
        "path": path,
        "exists": exists,
        "size_bytes": size,
        "size_kb": round(size / 1024, 1),
        "sha256": _sha256_file(path) if exists else "missing",
    }


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write JSON to ``path`` atomically; survives partial crashes."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[easy3e_demo] could not write report: {e}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        default="experiments/ultrashape/inputs/images/test_mug.png",
        help="Input image (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--instruction",
        default="paint the mug bright red with a glossy finish",
        help=(
            "Text instruction for the edit. Because the current editor "
            "stack drives edits via feature repainting under a mask, "
            "semantic/appearance edits (color, material, surface style) "
            "work better than topological edits (adding limbs, etc.)."
        ),
    )
    parser.add_argument(
        "--view",
        default="front",
        choices=["front", "back", "left", "right", "top", "bottom"],
        help="Which view to render + edit from.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/clearmesh_easy3e_demo",
        help="Where to write artifacts and report.",
    )
    parser.add_argument("--ultrashape-dir", default="/workspace/UltraShape-1.0")
    parser.add_argument(
        "--ultrashape-checkpoint",
        default="/workspace/checkpoints/ultrashape_v1.pt",
    )
    parser.add_argument("--trellis2-dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--model-dir", default="/workspace/models/trellis2-4b")
    parser.add_argument(
        "--resolution", type=int, default=512, choices=[512, 1024, 1536]
    )
    parser.add_argument("--octree-res", type=int, default=512, choices=[512, 1024])
    parser.add_argument(
        "--text-image-guidance",
        type=float,
        default=1.5,
        help="InstructPix2Pix image_guidance_scale (higher = preserve source).",
    )
    parser.add_argument(
        "--text-guidance-scale",
        type=float,
        default=7.5,
        help="InstructPix2Pix guidance_scale (higher = stronger instruction).",
    )
    parser.add_argument(
        "--text-num-steps", type=int, default=20, help="InstructPix2Pix diffusion steps."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the 2D image edit."
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help=(
            "Skip base mesh generation and reuse an existing "
            "``{output-dir}/base_mesh.glb`` — useful when iterating on "
            "the edit step."
        ),
    )
    args = parser.parse_args()

    repo_root = str(Path(__file__).resolve().parent.parent)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input path relative to repo root if not absolute.
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = str(Path(repo_root) / input_path)

    base_mesh_path = str(out_dir / "base_mesh.glb")
    source_render_path = str(out_dir / "source_render.png")
    target_image_path = str(out_dir / "target_image.png")
    edited_mesh_path = str(out_dir / "edited_mesh.glb")
    report_path = str(out_dir / "report.json")

    # ── Env fingerprint (reuse e2e_smoke's collect_env so both reports
    # share a schema) ──────────────────────────────────────────────────
    sys.path.insert(0, os.path.join(repo_root, "scripts"))
    from e2e_smoke import collect_env  # type: ignore

    report: dict = {
        "overall_pass": False,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "env": collect_env(repo_root, args.trellis2_dir),
        "args": vars(args),
        "instruction": args.instruction,
        "view": args.view,
        "paths": {
            "input": input_path,
            "base_mesh": base_mesh_path,
            "source_render": source_render_path,
            "target_image": target_image_path,
            "edited_mesh": edited_mesh_path,
            "report": report_path,
        },
        "stages": {},
        "artifacts": {},
        "errors": [],
    }

    def _snapshot():
        _atomic_write_json(report_path, report)

    # Preflight — input exists, UltraShape checkpoint exists.
    if not os.path.exists(input_path):
        report["stages"]["preflight_input"] = {
            "pass": False,
            "reason": f"input image not found: {input_path}",
        }
        _snapshot()
        print(f"[easy3e_demo] FAIL: input not found at {input_path}", file=sys.stderr)
        return 2
    report["stages"]["preflight_input"] = {"pass": True, "path": input_path}

    ckpt_exists = os.path.exists(args.ultrashape_checkpoint)
    report["stages"]["preflight_ultrashape_ckpt"] = {
        "pass": ckpt_exists,
        "path": args.ultrashape_checkpoint,
        "reason": None if ckpt_exists else "ultrashape checkpoint not found",
    }
    if not ckpt_exists:
        _snapshot()
        print(
            f"[easy3e_demo] FAIL: ultrashape checkpoint not found at "
            f"{args.ultrashape_checkpoint}",
            file=sys.stderr,
        )
        return 2

    # ── Stage 1: Base mesh (skip if user pre-generated one) ────────────
    if args.skip_base and os.path.exists(base_mesh_path):
        print(f"[easy3e_demo] --skip-base: reusing {base_mesh_path}")
        report["stages"]["base_mesh"] = {"pass": True, "skipped": True}
    else:
        print(
            f"[easy3e_demo] Stage 1: generating base mesh from {input_path} "
            f"(resolution={args.resolution}, octree_res={args.octree_res})"
        )
        t0 = time.time()
        try:
            from clearmesh.pipeline import ClearMeshPipeline, GenerationOptions

            pipeline = ClearMeshPipeline(
                ultrashape_dir=args.ultrashape_dir,
                ultrashape_checkpoint=args.ultrashape_checkpoint,
            )
            gen_options = GenerationOptions(
                resolution=args.resolution,
                refinement_octree_res=args.octree_res,
                export_format="glb",
            )
            gen_result = pipeline.generate(input_path, base_mesh_path, gen_options)
            duration = round(time.time() - t0, 2)
            report["stages"]["base_mesh"] = {
                "pass": True,
                "duration_s": duration,
                "substages": {
                    stage: round(dur, 3)
                    for stage, dur in (gen_result.timings or {}).items()
                },
            }
            print(f"[easy3e_demo]   base mesh ready in {duration}s")
        except Exception as e:
            report["stages"]["base_mesh"] = {
                "pass": False,
                "reason": str(e),
                "traceback": traceback.format_exc(),
            }
            report["errors"].append(f"base mesh generation failed: {e}")
            _snapshot()
            print(f"[easy3e_demo] FAIL in Stage 1: {e}", file=sys.stderr)
            return 1

    report["artifacts"]["base_mesh"] = _file_info(base_mesh_path)
    _snapshot()

    # ── Stage 2: Render source view + run InstructPix2Pix ──────────────
    # Composed explicitly (rather than calling edit_from_text directly)
    # so source_render.png + target_image.png land on disk for debugging.
    print(f"[easy3e_demo] Stage 2: rendering source view + InstructPix2Pix")
    print(f"[easy3e_demo]   instruction: {args.instruction!r}")
    t0 = time.time()
    try:
        from clearmesh.editing.image_edit import ImageEditor

        image_editor = ImageEditor()
        source_render = image_editor._render_view(
            base_mesh_path, args.view, image_size=512
        )
        source_render.save(source_render_path)
        print(f"[easy3e_demo]   source render saved to {source_render_path}")

        target_image = image_editor.edit(
            source_image=source_render,
            instruction=args.instruction,
            num_inference_steps=args.text_num_steps,
            image_guidance_scale=args.text_image_guidance,
            guidance_scale=args.text_guidance_scale,
            seed=args.seed,
        )
        target_image.save(target_image_path)
        duration = round(time.time() - t0, 2)
        report["stages"]["image_edit"] = {
            "pass": True,
            "duration_s": duration,
            "model": image_editor.model_id,
            "num_inference_steps": args.text_num_steps,
            "image_guidance_scale": args.text_image_guidance,
            "guidance_scale": args.text_guidance_scale,
        }
        print(f"[easy3e_demo]   target image ready in {duration}s ({target_image_path})")
    except Exception as e:
        report["stages"]["image_edit"] = {
            "pass": False,
            "reason": str(e),
            "traceback": traceback.format_exc(),
        }
        report["errors"].append(f"image edit failed: {e}")
        _snapshot()
        print(f"[easy3e_demo] FAIL in Stage 2: {e}", file=sys.stderr)
        return 1

    report["artifacts"]["source_render"] = _file_info(source_render_path)
    report["artifacts"]["target_image"] = _file_info(target_image_path)
    _snapshot()

    # ── Stage 3: Easy3E edit (SLAT encode → flow-edit → repaint → decode) ─
    print(f"[easy3e_demo] Stage 3: Easy3E edit")
    t0 = time.time()
    try:
        from clearmesh.editing import Easy3EEditor
        from clearmesh.editing.easy3e import EditOptions

        editor = Easy3EEditor(
            trellis2_dir=args.trellis2_dir,
            model_dir=args.model_dir,
        )
        # Record whether the strict voxel-flow path was loaded. This is
        # the single most informative bit for debugging edit quality: if
        # flow_model_loaded=False, the edit is feature-repaint-only and
        # topological changes won't land. Captured *before* the edit so
        # we see it even on failure.
        _ = editor.voxel_flowedit  # force lazy-load
        flow_model_loaded = editor.voxel_flowedit.flow_model is not None
        report["stages"]["edit_flow_model_loaded"] = {
            "pass": True,
            "value": flow_model_loaded,
            "fingerprint": editor._build_fingerprint(),
        }

        edit_options = EditOptions(
            num_flow_steps=25,
            num_repaint_steps=25,
            text_num_steps=args.text_num_steps,
            text_image_guidance=args.text_image_guidance,
            text_guidance_scale=args.text_guidance_scale,
            enable_texture=False,  # Ctrl-Adapter path is not wired yet
            enable_repair=True,
            export_format="glb",
        )

        # Compose explicitly rather than edit_from_text so we reuse the
        # source_render + target_image we already saved to disk.
        edit_result = editor.edit(
            source_mesh=base_mesh_path,
            edit_image=target_image_path,
            source_image=source_render_path,
            output_path=edited_mesh_path,
            options=edit_options,
        )
        duration = round(time.time() - t0, 2)
        report["stages"]["easy3e_edit"] = {
            "pass": True,
            "duration_s": duration,
            "substages": {
                stage: round(dur, 3)
                for stage, dur in (edit_result.timings or {}).items()
            },
        }

        # Mesh sanity on the edited result.
        try:
            m = edit_result.mesh
            report["edited_mesh_stats"] = {
                "vertices": int(m.vertices.shape[0]),
                "faces": int(m.faces.shape[0]),
                "watertight": bool(m.is_watertight),
                "volume": (
                    float(m.volume) if getattr(m, "is_watertight", False) else None
                ),
            }
        except Exception as e:
            report["errors"].append(f"edited mesh stats failed: {e}")

        print(f"[easy3e_demo]   edit complete in {duration}s")
    except Exception as e:
        report["stages"]["easy3e_edit"] = {
            "pass": False,
            "reason": str(e),
            "traceback": traceback.format_exc(),
        }
        report["errors"].append(f"easy3e edit failed: {e}")
        _snapshot()
        print(f"[easy3e_demo] FAIL in Stage 3: {e}", file=sys.stderr)
        return 1

    report["artifacts"]["edited_mesh"] = _file_info(edited_mesh_path)

    # Overall pass: every stage green + every artifact present +
    # edited mesh is non-degenerate.
    stages_ok = all(s.get("pass", False) for s in report["stages"].values())
    artifacts_ok = all(a.get("exists", False) for a in report["artifacts"].values())
    mesh_ok = report.get("edited_mesh_stats", {}).get("vertices", 0) > 0
    report["overall_pass"] = stages_ok and artifacts_ok and mesh_ok
    _snapshot()

    verdict = "PASS" if report["overall_pass"] else "FAIL"
    print(
        f"\n[easy3e_demo] {verdict}  "
        f"base={Path(base_mesh_path).name} ({report['artifacts']['base_mesh']['size_kb']} KB)  "
        f"edited={Path(edited_mesh_path).name} ({report['artifacts']['edited_mesh']['size_kb']} KB)  "
        f"verts={report.get('edited_mesh_stats', {}).get('vertices', 'n/a')}  "
        f"report={report_path}"
    )
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
