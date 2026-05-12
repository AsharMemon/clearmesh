"""Blender/DCC promotion gates for production mesh assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


@dataclass(frozen=True)
class BlenderGateOptions:
    blender: str = "blender"
    timeout_seconds: int = 180


@dataclass
class BlenderGateReport:
    mesh_path: str
    ok: bool
    skipped: bool
    reason: str | None
    blender: str | None
    import_ok: bool | None = None
    roundtrip_ok: bool | None = None
    subdivision_ok: bool | None = None
    deformation_smoke_ok: bool | None = None
    mesh_object_count: int | None = None
    vertex_count: int | None = None
    face_count: int | None = None
    quad_ratio: float | None = None
    output_glb: str | None = None
    stdout_tail: str | None = None
    stderr_tail: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_blender_mesh_gates(
    mesh_path: str | Path,
    output_dir: str | Path,
    options: BlenderGateOptions | None = None,
) -> BlenderGateReport:
    options = options or BlenderGateOptions()
    mesh_path = Path(mesh_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blender_path = _resolve_blender(options.blender)
    if blender_path is None:
        return BlenderGateReport(
            mesh_path=str(mesh_path),
            ok=False,
            skipped=True,
            reason="blender_not_found",
            blender=None,
        )
    if not mesh_path.exists():
        return BlenderGateReport(
            mesh_path=str(mesh_path),
            ok=False,
            skipped=False,
            reason="mesh_not_found",
            blender=blender_path,
        )

    result_json = output_dir / "blender_gates.raw.json"
    roundtrip_glb = output_dir / "blender_roundtrip.glb"
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "blender_gates.py"
        script.write_text(_BLENDER_SCRIPT, encoding="utf-8")
        result = subprocess.run(
            [
                blender_path,
                "--background",
                "--python",
                str(script),
                "--",
                str(mesh_path),
                str(roundtrip_glb),
                str(result_json),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(options.timeout_seconds)),
        )
    stdout_tail = (result.stdout or "")[-2000:]
    stderr_tail = (result.stderr or "")[-2000:]
    if result.returncode != 0:
        return BlenderGateReport(
            mesh_path=str(mesh_path),
            ok=False,
            skipped=False,
            reason=f"blender_failed:{result.returncode}",
            blender=blender_path,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
    try:
        details = json.loads(result_json.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - gate should report failure, not crash the worker.
        return BlenderGateReport(
            mesh_path=str(mesh_path),
            ok=False,
            skipped=False,
            reason=f"invalid_gate_json:{type(exc).__name__}",
            blender=blender_path,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
    ok = bool(
        details.get("import_ok")
        and details.get("roundtrip_ok")
        and details.get("subdivision_ok")
        and details.get("deformation_smoke_ok")
    )
    return BlenderGateReport(
        mesh_path=str(mesh_path),
        ok=ok,
        skipped=False,
        reason=None if ok else "one_or_more_blender_gates_failed",
        blender=blender_path,
        import_ok=_optional_bool(details.get("import_ok")),
        roundtrip_ok=_optional_bool(details.get("roundtrip_ok")),
        subdivision_ok=_optional_bool(details.get("subdivision_ok")),
        deformation_smoke_ok=_optional_bool(details.get("deformation_smoke_ok")),
        mesh_object_count=_optional_int(details.get("mesh_object_count")),
        vertex_count=_optional_int(details.get("vertex_count")),
        face_count=_optional_int(details.get("face_count")),
        quad_ratio=_optional_float(details.get("quad_ratio")),
        output_glb=str(roundtrip_glb) if roundtrip_glb.exists() else None,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        details=details,
    )


def write_blender_gate_report(path: str | Path, report: BlenderGateReport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _resolve_blender(value: str) -> str | None:
    found = shutil.which(value)
    if found:
        return found
    path = Path(value).expanduser()
    if path.exists():
        return str(path)
    return None


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


_BLENDER_SCRIPT = r'''
import json
from pathlib import Path
import sys

import bpy


src, dst, report_path = sys.argv[-3], sys.argv[-2], sys.argv[-1]
report = {
    "import_ok": False,
    "roundtrip_ok": False,
    "subdivision_ok": False,
    "deformation_smoke_ok": False,
}


def import_mesh(path):
    lower = path.lower()
    if lower.endswith(".obj"):
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    elif lower.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    elif lower.endswith(".stl"):
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=path)
        else:
            bpy.ops.import_mesh.stl(filepath=path)
    else:
        raise RuntimeError("unsupported Blender import format")


def active_mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def evaluate_modifier(obj, modifier_type, configure=None):
    duplicate = obj.copy()
    duplicate.data = obj.data.copy()
    bpy.context.collection.objects.link(duplicate)
    bpy.context.view_layer.objects.active = duplicate
    duplicate.select_set(True)
    try:
        modifier = duplicate.modifiers.new(name="clearmesh_gate", type=modifier_type)
        if configure is not None:
            configure(modifier)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = duplicate.evaluated_get(depsgraph)
        _ = len(evaluated.to_mesh().polygons)
        return True
    finally:
        bpy.data.objects.remove(duplicate, do_unlink=True)


try:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    import_mesh(src)
    meshes = active_mesh_objects()
    report["mesh_object_count"] = len(meshes)
    if not meshes:
        raise RuntimeError("no mesh objects imported")
    report["import_ok"] = True
    vertex_count = 0
    face_count = 0
    quad_count = 0
    for obj in meshes:
        mesh = obj.data
        vertex_count += len(mesh.vertices)
        face_count += len(mesh.polygons)
        quad_count += sum(1 for poly in mesh.polygons if len(poly.vertices) == 4)
    report["vertex_count"] = vertex_count
    report["face_count"] = face_count
    report["quad_ratio"] = quad_count / max(face_count, 1)
    primary = max(meshes, key=lambda obj: len(obj.data.polygons))
    report["subdivision_ok"] = evaluate_modifier(
        primary,
        "SUBSURF",
        lambda modifier: (setattr(modifier, "levels", 1), setattr(modifier, "render_levels", 1)),
    )
    report["deformation_smoke_ok"] = evaluate_modifier(
        primary,
        "SIMPLE_DEFORM",
        lambda modifier: (setattr(modifier, "deform_method", "BEND"), setattr(modifier, "angle", 0.05)),
    )
    bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
    report["roundtrip_ok"] = Path(dst).exists()
except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"

Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
'''
