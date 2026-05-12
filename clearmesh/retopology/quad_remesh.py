"""Optional quad remeshing baselines for ClearMesh.

This module is a benchmark lane, not a universal artist-topology solver. It
prefers Instant Meshes bindings when installed, can call an Instant Meshes CLI,
and otherwise emits a deterministic quad cage for smoke tests and template work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import trimesh

from clearmesh.eval.mesh_quality import evaluate_mesh, load_mesh


@dataclass(frozen=True)
class QuadRemeshOptions:
    engine: str = "auto"
    target_faces: int = 5_000
    target_vertices: int = -1
    pure_quad: bool = True
    deterministic: bool = True
    crease_angle: float = -1.0
    align_to_boundaries: bool = True
    smooth_iterations: int = 2
    instant_meshes_path: str | None = None
    quadriflow_path: str | None = None
    quadriflow_sharp: bool = True
    quadriflow_mcf: bool = False
    cage_subdivisions: int | None = None
    weld_tolerance: float = 1e-8
    measure_weld_metrics: bool = True


@dataclass
class QuadRemeshReport:
    input_path: str
    output_path: str
    engine: str
    target_faces: int
    target_vertices: int
    quad_stats: dict[str, Any]
    pre_weld_metrics: dict[str, Any]
    mesh_metrics: dict[str, Any]
    postprocess: dict[str, Any]
    weld_effect: dict[str, Any]
    notes: list[str]


def quad_remesh_file(
    input_path: str | Path,
    output_path: str | Path,
    options: QuadRemeshOptions | None = None,
) -> QuadRemeshReport:
    options = options or QuadRemeshOptions()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    engine = options.engine
    if engine == "auto":
        if _pyinstantmeshes_available():
            engine = "pyinstantmeshes"
        elif _instant_meshes_binary(options.instant_meshes_path):
            engine = "instant_meshes_cli"
        elif _quadriflow_binary(options.quadriflow_path):
            engine = "quadriflow_cli"
        else:
            engine = "template_cage"

    if engine == "pyinstantmeshes":
        try:
            _run_pyinstantmeshes(input_path, output_path, options)
        except Exception as exc:  # noqa: BLE001 - fallback keeps benchmark lane usable.
            notes.append(f"pyinstantmeshes failed: {type(exc).__name__}: {exc}")
            engine = "template_cage"
            _run_template_cage(input_path, output_path, options)
    elif engine == "instant_meshes_cli":
        _run_instant_meshes_cli(input_path, output_path, options)
    elif engine == "quadriflow_cli":
        _run_quadriflow_cli(input_path, output_path, options)
    elif engine == "template_cage":
        _run_template_cage(input_path, output_path, options)
    else:
        raise ValueError(f"unsupported quad remesh engine: {engine}")

    pre_weld_metrics = evaluate_mesh(output_path) if options.measure_weld_metrics else {}
    postprocess = weld_obj_vertices(output_path, tolerance=options.weld_tolerance)
    post_weld_metrics = evaluate_mesh(output_path)

    return QuadRemeshReport(
        input_path=str(input_path),
        output_path=str(output_path),
        engine=engine,
        target_faces=options.target_faces,
        target_vertices=options.target_vertices,
        quad_stats=quad_mesh_stats(output_path),
        pre_weld_metrics=pre_weld_metrics,
        mesh_metrics=post_weld_metrics,
        postprocess=postprocess,
        weld_effect=summarize_weld_effect(pre_weld_metrics, post_weld_metrics, postprocess),
        notes=notes,
    )


def quad_remesh_options_from_metadata(metadata: dict[str, Any]) -> QuadRemeshOptions:
    return QuadRemeshOptions(
        engine=str(metadata.get("quad_remesh_engine", "auto")),
        target_faces=int(metadata.get("quad_target_faces", 5_000)),
        target_vertices=int(metadata.get("quad_target_vertices", -1)),
        pure_quad=_bool(metadata.get("quad_pure", True)),
        deterministic=_bool(metadata.get("quad_deterministic", True)),
        crease_angle=float(metadata.get("quad_crease_angle", -1.0)),
        align_to_boundaries=_bool(metadata.get("quad_align_to_boundaries", True)),
        smooth_iterations=int(metadata.get("quad_smooth_iterations", 2)),
        instant_meshes_path=_optional_str(metadata.get("instant_meshes_path")),
        quadriflow_path=_optional_str(metadata.get("quadriflow_path")),
        quadriflow_sharp=_bool(metadata.get("quadriflow_sharp", metadata.get("quad_preserve_sharp", True))),
        quadriflow_mcf=_bool(metadata.get("quadriflow_mcf", False)),
        cage_subdivisions=_optional_int(metadata.get("quad_cage_subdivisions")),
        weld_tolerance=float(metadata.get("quad_weld_tolerance", 1e-8)),
        measure_weld_metrics=_bool(metadata.get("quad_measure_weld_metrics", True)),
    )


def quad_mesh_stats(path: str | Path) -> dict[str, Any]:
    face_sizes = _obj_face_sizes(path)
    total = max(len(face_sizes), 1)
    quad_count = sum(1 for size in face_sizes if size == 4)
    tri_count = sum(1 for size in face_sizes if size == 3)
    ngon_count = sum(1 for size in face_sizes if size not in {3, 4})
    return {
        "face_count": len(face_sizes),
        "quad_count": quad_count,
        "tri_count": tri_count,
        "ngon_count": ngon_count,
        "quad_ratio": quad_count / total,
        "pure_quad": bool(face_sizes and quad_count == len(face_sizes)),
    }


def report_to_dict(report: QuadRemeshReport) -> dict[str, Any]:
    return asdict(report)


def summarize_weld_effect(before: dict[str, Any], after: dict[str, Any], postprocess: dict[str, Any]) -> dict[str, Any]:
    if not before or not after or not before.get("ok") or not after.get("ok"):
        return {"ok": False, "reason": "missing_metrics"}
    fields = [
        "vertex_count",
        "face_count",
        "connected_components",
        "tiny_component_count",
        "boundary_loop_count",
        "nonmanifold_edge_count",
        "nonmanifold_vertex_count",
    ]
    deltas: dict[str, int | None] = {}
    for field in fields:
        before_value = before.get(field)
        after_value = after.get(field)
        deltas[field] = None if before_value is None or after_value is None else int(after_value) - int(before_value)

    before_component_raw = int(before.get("connected_components") or 0)
    before_components = max(before_component_raw, 1)
    after_components = int(after.get("connected_components") or 0)
    before_boundary_raw = int(before.get("boundary_loop_count") or 0)
    before_boundary = max(before_boundary_raw, 1)
    after_boundary = int(after.get("boundary_loop_count") or 0)
    component_reduction = 1.0 - (after_components / before_components)
    boundary_reduction = 0.0 if before_boundary_raw == 0 else 1.0 - (after_boundary / before_boundary)
    status = "no_op"
    if component_reduction > 0.8 and after_components <= 64:
        status = "strong_weld"
    elif component_reduction > 0.5 or boundary_reduction > 0.5:
        status = "helped_but_recheck"
    elif int(postprocess.get("vertices_before", 0) or 0) != int(postprocess.get("vertices_after", 0) or 0):
        status = "geometry_welded_without_topology_gain"
    return {
        "ok": True,
        "status": status,
        "component_reduction": float(component_reduction),
        "boundary_loop_reduction": float(boundary_reduction),
        "deltas": deltas,
        "promotion_hint": "do_not_promote" if after_components > 64 or after_boundary > 128 else "eligible_for_deeper_quad_gates",
    }


def weld_obj_vertices(path: str | Path, tolerance: float = 1e-8) -> dict[str, Any]:
    """Merge coincident OBJ vertices while preserving polygon face sizes.

    Instant Meshes style tools can emit visually correct quads with duplicated
    boundary vertices. DCC tools then treat those quads as disconnected islands.
    This pass rewrites the OBJ as position-only `v`/`f` data with shared vertex
    indices so topology metrics reflect the editable surface an artist expects.
    """

    path = Path(path)
    if tolerance <= 0 or path.suffix.lower() != ".obj" or not path.exists():
        return {"enabled": False}

    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                face: list[int] = []
                for token in line.strip().split()[1:]:
                    raw = token.split("/", 1)[0]
                    if not raw:
                        continue
                    index = int(raw)
                    if index < 0:
                        index = len(vertices) + index + 1
                    face.append(index - 1)
                if face:
                    faces.append(face)

    if not vertices or not faces:
        return {"enabled": True, "vertices_before": len(vertices), "faces_before": len(faces), "skipped": "empty_obj"}

    unique_vertices: list[tuple[float, float, float]] = []
    key_to_index: dict[tuple[int, int, int], int] = {}
    remap: list[int] = []
    for vertex in vertices:
        key = tuple(int(round(coord / tolerance)) for coord in vertex)
        if key not in key_to_index:
            key_to_index[key] = len(unique_vertices)
            unique_vertices.append(vertex)
        remap.append(key_to_index[key])

    welded_faces: list[list[int]] = []
    removed_degenerate = 0
    for face in faces:
        welded = [remap[index] for index in face if 0 <= index < len(remap)]
        if len(set(welded)) < 3:
            removed_degenerate += 1
            continue
        welded_faces.append(welded)

    if len(unique_vertices) == len(vertices) and removed_degenerate == 0:
        return {
            "enabled": True,
            "vertices_before": len(vertices),
            "vertices_after": len(vertices),
            "faces_before": len(faces),
            "faces_after": len(faces),
            "removed_degenerate_faces": 0,
        }

    _write_obj(path, np.asarray(unique_vertices, dtype=float), np.asarray(welded_faces, dtype=object))
    return {
        "enabled": True,
        "vertices_before": len(vertices),
        "vertices_after": len(unique_vertices),
        "faces_before": len(faces),
        "faces_after": len(welded_faces),
        "removed_degenerate_faces": removed_degenerate,
    }


def _run_pyinstantmeshes(input_path: Path, output_path: Path, options: QuadRemeshOptions) -> None:
    import pyinstantmeshes  # type: ignore

    kwargs = {
        "target_vertex_count": options.target_vertices,
        "target_face_count": options.target_faces,
        "rosy": 4,
        "posy": 4,
        "crease_angle": options.crease_angle,
        "align_to_boundaries": options.align_to_boundaries,
        "smooth_iterations": options.smooth_iterations,
        "pure_quad": options.pure_quad,
        "deterministic": options.deterministic,
    }
    file_api_suffixes = {".obj", ".ply", ".aln"}
    if hasattr(pyinstantmeshes, "remesh_file") and input_path.suffix.lower() in file_api_suffixes:
        try:
            result = pyinstantmeshes.remesh_file(str(input_path), str(output_path), **kwargs)
            if output_path.exists():
                return
            if isinstance(result, tuple) and len(result) == 2:
                _write_obj(output_path, result[0], result[1])
                return
        except Exception:
            if not hasattr(pyinstantmeshes, "remesh"):
                raise

    mesh = load_mesh(input_path)
    vertices, faces = pyinstantmeshes.remesh(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        **kwargs,
    )
    _write_obj(output_path, vertices, faces)


def _run_instant_meshes_cli(input_path: Path, output_path: Path, options: QuadRemeshOptions) -> None:
    binary = _instant_meshes_binary(options.instant_meshes_path)
    if binary is None:
        raise FileNotFoundError("Instant Meshes binary not found")
    with tempfile.TemporaryDirectory(prefix="clearmesh_instant_meshes_") as tmp:
        remesher_input = _external_remesher_input(input_path, Path(tmp))
        command = [
            binary,
            "-o",
            str(output_path),
            "-f",
            str(options.target_faces),
            "-r",
            "4",
            "-p",
            "4",
            "-S",
            str(options.smooth_iterations),
        ]
        if options.deterministic:
            command.append("-d")
        if not options.pure_quad:
            command.append("-D")
        if options.align_to_boundaries:
            command.append("-b")
        if options.crease_angle >= 0:
            command.extend(["-c", str(options.crease_angle)])
        command.append(str(remesher_input))
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _run_quadriflow_cli(input_path: Path, output_path: Path, options: QuadRemeshOptions) -> None:
    binary = _quadriflow_binary(options.quadriflow_path)
    if binary is None:
        raise FileNotFoundError("QuadriFlow binary not found")
    with tempfile.TemporaryDirectory(prefix="clearmesh_quadriflow_") as tmp:
        remesher_input = _external_remesher_input(input_path, Path(tmp))
        command = [binary]
        if options.quadriflow_mcf:
            command.append("-mcf")
        if options.quadriflow_sharp:
            command.append("-sharp")
        command.extend(["-i", str(remesher_input), "-o", str(output_path), "-f", str(options.target_faces)])
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _run_template_cage(input_path: Path, output_path: Path, options: QuadRemeshOptions) -> None:
    mesh = load_mesh(input_path)
    subdivisions = options.cage_subdivisions or max(1, int(round(np.sqrt(max(options.target_faces, 6) / 6))))
    vertices, quads = _subdivided_box(mesh.bounds, subdivisions=subdivisions)
    _write_obj(output_path, vertices, quads)


def _external_remesher_input(input_path: Path, tmp_dir: Path) -> Path:
    """Materialize unsupported mesh containers as OBJ for external remeshers."""

    if input_path.suffix.lower() in {".obj", ".ply"}:
        return input_path
    mesh = load_mesh(input_path)
    converted = tmp_dir / "source.obj"
    mesh.export(converted)
    return converted


def _subdivided_box(bounds: np.ndarray, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        bounds = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=float)
    lo, hi = bounds
    if np.linalg.norm(hi - lo) < 1e-9:
        lo = lo - 0.5
        hi = hi + 0.5
    subdivisions = max(1, int(subdivisions))
    vertices: list[tuple[float, float, float]] = []
    index: dict[tuple[float, float, float], int] = {}
    quads: list[list[int]] = []

    def add_vertex(point: tuple[float, float, float]) -> int:
        key = tuple(round(float(value), 10) for value in point)
        if key not in index:
            index[key] = len(vertices)
            vertices.append(key)
        return index[key]

    def add_face(axis: int, value: float, reverse: bool) -> None:
        axes = [0, 1, 2]
        axes.remove(axis)
        a0, a1 = axes
        for i in range(subdivisions):
            for j in range(subdivisions):
                coords = []
                for u, v in (
                    (i, j),
                    (i + 1, j),
                    (i + 1, j + 1),
                    (i, j + 1),
                ):
                    point = [0.0, 0.0, 0.0]
                    point[axis] = value
                    point[a0] = lo[a0] + (hi[a0] - lo[a0]) * (u / subdivisions)
                    point[a1] = lo[a1] + (hi[a1] - lo[a1]) * (v / subdivisions)
                    coords.append(add_vertex(tuple(point)))
                quads.append(list(reversed(coords)) if reverse else coords)

    add_face(0, lo[0], True)
    add_face(0, hi[0], False)
    add_face(1, lo[1], False)
    add_face(1, hi[1], True)
    add_face(2, lo[2], True)
    add_face(2, hi[2], False)
    return np.asarray(vertices, dtype=float), np.asarray(quads, dtype=np.int64)


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=float)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# ClearMesh quad remesh output\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in faces:
            face = [int(index) + 1 for index in face if int(index) >= 0]
            handle.write("f " + " ".join(str(index) for index in face) + "\n")


def _obj_face_sizes(path: str | Path) -> list[int]:
    sizes: list[int] = []
    try:
        with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("f "):
                    sizes.append(len(line.strip().split()) - 1)
    except UnicodeDecodeError:
        return []
    return sizes


def _pyinstantmeshes_available() -> bool:
    try:
        import pyinstantmeshes  # noqa: F401

        return True
    except Exception:
        return False


def _instant_meshes_binary(configured: str | None) -> str | None:
    candidates = [configured] if configured else []
    candidates.extend(["InstantMeshes", "instant-meshes", "instantmeshes"])
    for candidate in candidates:
        if not candidate:
            continue
        path = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if path:
            return str(path)
    return None


def _quadriflow_binary(configured: str | None) -> str | None:
    candidates = [configured] if configured else []
    candidates.extend(["quadriflow", "QuadriFlow"])
    for candidate in candidates:
        if not candidate:
            continue
        path = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if path:
            return str(path)
    return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
