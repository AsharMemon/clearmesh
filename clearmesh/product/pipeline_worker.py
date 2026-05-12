"""Production worker primitives for ClearMesh jobs.

The worker is intentionally staged. It can run fully deterministic bookkeeping on
any machine, then enable GPU-heavy steps once TRELLIS.2 and mesh-head repos are
installed on a GPU host.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from .artifacts import ArtifactStore
from .commands import discover_new_mesh, run_logged_command
from .jobs import JobService
from .models import AssetRecord, JobRecord, JobStatus, JobStepStatus
from .store import JsonJobStore


class PipelineWorker:
    def __init__(
        self,
        *,
        store: JsonJobStore,
        service: JobService,
        artifacts: ArtifactStore,
        execute_heavy: bool = False,
        mesh_head: str = "meshripple",
        mesh_head_config: dict | None = None,
        preferred_point_budget: int = 40960,
    ) -> None:
        self.store = store
        self.service = service
        self.artifacts = artifacts
        self.execute_heavy = execute_heavy
        self.mesh_head = mesh_head
        self.mesh_head_config = mesh_head_config or {}
        self.preferred_point_budget = preferred_point_budget

    def run_once(self) -> str | None:
        job = self.store.claim_next_queued()
        if job is None:
            return None
        self.run_job(job.id, already_claimed=True)
        return job.id

    def run_job(self, job_id: str, *, already_claimed: bool = False) -> JobRecord:
        if not already_claimed:
            self.store.set_status(job_id, JobStatus.RUNNING)
        try:
            self._run_input_validation(job_id)
            edited_mesh = self._run_easy3e_edit(job_id)
            proxy_mesh = edited_mesh or self._run_trellis_proxy(job_id)
            proxy_mesh = self._run_coarse_adapter(job_id, proxy_mesh)
            reference_mesh = self._run_reference_refinement(job_id, proxy_mesh)
            proxy_mesh = reference_mesh or proxy_mesh
            self._run_mesh_passport(job_id, proxy_mesh)
            control_mesh = self._run_surface_normalization(job_id, proxy_mesh)
            control_mesh = self._run_shrinkwrap_projection(job_id, control_mesh, proxy_mesh)
            self._run_retopology_planning(job_id, control_mesh)
            self._run_chart_remesh(job_id, control_mesh)
            chart_stitched_mesh = self._run_chart_stitch(job_id)
            quad_mesh = self._run_quad_remesh(job_id, control_mesh)
            projection_source = chart_stitched_mesh if chart_stitched_mesh and self._metadata_bool(job_id, "chart_stitch_prefer_for_projection", default=False) else quad_mesh
            projected_quad = self._run_feature_projection(job_id, projection_source, proxy_mesh)
            if projected_quad and self._metadata_bool(job_id, "feature_projection_as_final", default=False):
                control_mesh = projected_quad
            elif chart_stitched_mesh and self._metadata_bool(job_id, "chart_stitch_as_final", default=False):
                control_mesh = chart_stitched_mesh
            elif quad_mesh and self._metadata_bool(job_id, "quad_remesh_as_final", default=False):
                control_mesh = quad_mesh
            self._run_preview_publish(job_id, control_mesh)
            point_cloud = self._run_point_cloud_bridge(job_id, control_mesh)
            self._run_part_structure(job_id)
            mesh_path = self._run_mesh_head(job_id, point_cloud, control_mesh)
            mesh_path = mesh_path or control_mesh
            mesh_path = self._run_mesh_cleanup(job_id, mesh_path)
            self._run_repair_validation(job_id, mesh_path)
            self._run_production_gate(job_id, mesh_path)
            self._run_export_package(job_id, mesh_path)
            job = self.store.get_job(job_id)
            if any(step.name == "autorigging" for step in job.steps):
                self._run_autorigging(job_id, mesh_path)
            return self.service.mark_succeeded(job_id)
        except Exception as exc:  # noqa: BLE001 - workers should persist failures.
            self._fail_running_steps(job_id, f"{type(exc).__name__}: {exc}")
            return self.service.mark_failed(job_id, error=f"{type(exc).__name__}: {exc}")

    def _job_dirs(self, job_id: str) -> tuple[JobRecord, str, Path]:
        job = self.store.get_job(job_id)
        project_id = str(job.request.metadata.get("project_id", "default"))
        root = self.artifacts.path(project_id, job_id, "")
        root.mkdir(parents=True, exist_ok=True)
        return job, project_id, root

    def _complete_step(self, job_id: str, name: str, artifacts: dict[str, str] | None = None) -> None:
        self.store.update_step(job_id, name, JobStepStatus.SUCCEEDED, artifacts=artifacts or {})

    def _skip_step(self, job_id: str, name: str, reason: str, artifacts: dict[str, str] | None = None) -> None:
        payload = dict(artifacts or {})
        payload["reason"] = reason
        self.store.update_step(job_id, name, JobStepStatus.SKIPPED, artifacts=payload)

    def _fail_running_steps(self, job_id: str, error: str) -> None:
        job = self.store.get_job(job_id)
        for step in job.steps:
            if step.status == JobStepStatus.RUNNING:
                self.store.update_step(job_id, step.name, JobStepStatus.FAILED, error=error)

    def _run_input_validation(self, job_id: str) -> None:
        job, _, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "input_validation", JobStepStatus.RUNNING)
        report = root / "reports" / "input_validation.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        input_path = self.artifacts.resolve_local_uri(job.request.input_uri)
        if input_path is not None and not input_path.exists():
            raise FileNotFoundError(f"resolved input_uri does not exist: {input_path}")
        report.write_text(
            f"input_uri={job.request.input_uri}\ninput_path={input_path or ''}\nmode={job.request.mode}\n",
            encoding="utf-8",
        )
        artifacts = {"report": str(report)}
        if input_path is not None:
            artifacts["input_path"] = str(input_path)
        self._complete_step(job_id, "input_validation", artifacts)

    def _run_trellis_proxy(self, job_id: str) -> Path | None:
        job, _, root = self._job_dirs(job_id)
        if job.status == JobStatus.CANCELED:
            raise RuntimeError("job was canceled")
        self.store.update_step(job_id, "trellis_proxy", JobStepStatus.RUNNING)
        proxy = job.request.metadata.get("proxy_mesh_path")
        if proxy:
            proxy_path = Path(str(proxy)).expanduser()
            if not proxy_path.exists():
                raise FileNotFoundError(f"proxy_mesh_path does not exist: {proxy_path}")
            self._complete_step(job_id, "trellis_proxy", {"proxy_mesh": str(proxy_path)})
            return proxy_path
        if not self.execute_heavy:
            self._skip_step(job_id, "trellis_proxy", "No proxy_mesh_path supplied and execute_heavy is false")
            return None
        command = job.request.metadata.get("trellis_command")
        if command:
            output_dir = root / "trellis_proxy"
            output_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = root / "logs" / "trellis_proxy.stdout.log"
            stderr_path = root / "logs" / "trellis_proxy.stderr.log"
            before = time.time()
            rendered = run_logged_command(
                command,
                cwd=job.request.metadata.get("trellis_cwd"),
                values=self._command_values(job, root, extra={"output_dir": str(output_dir)}),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                env=job.request.metadata.get("trellis_env"),
                timeout_seconds=int(job.request.metadata.get("trellis_timeout_seconds", 7200)),
            )
            proxy_path = discover_new_mesh(output_dir, since_mtime=before)
            if proxy_path is None:
                proxy_path = discover_new_mesh(root, since_mtime=before)
            if proxy_path is None:
                raise RuntimeError(f"TRELLIS command completed but no mesh was found under {output_dir}")
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="trellis_proxy_mesh",
                    uri=str(proxy_path),
                    content_type="model/mesh",
                    metadata={"command": rendered, "stdout": str(stdout_path), "stderr": str(stderr_path)},
                ),
            )
            self._complete_step(
                job_id,
                "trellis_proxy",
                {"proxy_mesh": str(proxy_path), "stdout": str(stdout_path), "stderr": str(stderr_path)},
            )
            return proxy_path
        report = root / "reports" / "trellis_proxy.todo.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("Supply metadata.proxy_mesh_path or metadata.trellis_command for GPU execution.\n", encoding="utf-8")
        self._skip_step(job_id, "trellis_proxy", "TRELLIS.2 command hook not configured", {"report": str(report)})
        return None

    def _run_coarse_adapter(self, job_id: str, proxy_mesh: Path | None) -> Path | None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "coarse_adapter", JobStepStatus.RUNNING)
        if proxy_mesh is None:
            self._skip_step(job_id, "coarse_adapter", "No proxy mesh available")
            return None
        if not self.execute_heavy:
            self._skip_step(job_id, "coarse_adapter", "execute_heavy is false", {"mesh": str(proxy_mesh)})
            return proxy_mesh
        if not self._metadata_bool(job_id, "coarse_adapter_enabled", default=False):
            self._skip_step(job_id, "coarse_adapter", "coarse_adapter_enabled is false", {"mesh": str(proxy_mesh)})
            return proxy_mesh

        supplied = job.request.metadata.get("coarse_adapter_mesh_path") or job.request.metadata.get("coarse_proxy_mesh_path")
        if supplied:
            adapted_path = Path(str(supplied)).expanduser()
            if not adapted_path.exists():
                raise FileNotFoundError(f"coarse adapter mesh does not exist: {adapted_path}")
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="coarse_proxy_mesh",
                    uri=str(adapted_path),
                    content_type="model/mesh",
                    metadata={"source": str(proxy_mesh), "stage": "metadata_supplied"},
                ),
            )
            self._complete_step(
                job_id,
                "coarse_adapter",
                {"mesh": str(adapted_path), "source_mesh": str(proxy_mesh), "engine": "metadata", "accepted": "true"},
            )
            return adapted_path

        from dataclasses import asdict

        from clearmesh.mesh.coarse_adapter import adapt_coarse_mesh_file, coarse_adapter_options_from_metadata

        output_dir = self.artifacts.path(project_id, job_id, "coarse_adapter", "")
        suffix = str(job.request.metadata.get("coarse_adapter_output_suffix", Path(proxy_mesh).suffix or ".glb"))
        adapted_path = output_dir / f"{Path(proxy_mesh).stem}_coarse{suffix}"
        report_path = root / "reports" / "coarse_adapter.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        options = coarse_adapter_options_from_metadata(job.request.metadata)
        report = adapt_coarse_mesh_file(proxy_mesh, adapted_path, options)
        report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")

        if not report.accepted and self._metadata_bool(job_id, "coarse_adapter_strict", default=False):
            raise RuntimeError(f"coarse adapter failed acceptance contract; see {report_path}")

        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="coarse_proxy_mesh",
                uri=str(adapted_path),
                content_type="model/mesh",
                metadata={
                    "source": str(proxy_mesh),
                    "engine": report.engine,
                    "accepted": str(report.accepted).lower(),
                    "report": str(report_path),
                },
            ),
        )
        self._complete_step(
            job_id,
            "coarse_adapter",
            {
                "mesh": str(adapted_path),
                "source_mesh": str(proxy_mesh),
                "engine": report.engine,
                "accepted": str(report.accepted).lower(),
                "report": str(report_path),
            },
        )
        return adapted_path

    def _run_mesh_passport(self, job_id: str, proxy_mesh: Path | None) -> None:
        _, _, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "mesh_passport", JobStepStatus.RUNNING)
        report_path = root / "reports" / "mesh_passport.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if proxy_mesh is None:
            report_path.write_text(
                json.dumps({"ok": False, "reason": "No proxy/control mesh available"}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self._skip_step(job_id, "mesh_passport", "No proxy mesh available", {"report": str(report_path)})
            return
        if not self.execute_heavy:
            report_path.write_text(
                json.dumps({"ok": False, "reason": "execute_heavy is false", "mesh": str(proxy_mesh)}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self._skip_step(job_id, "mesh_passport", "execute_heavy is false", {"report": str(report_path), "mesh": str(proxy_mesh)})
            return

        from clearmesh.product.mesh_passport import create_mesh_passport

        passport = create_mesh_passport(proxy_mesh)
        report_path.write_text(json.dumps(passport.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="mesh_passport",
                uri=str(report_path),
                content_type="application/json",
                metadata={"risk_level": passport.risk_level, "recommended_operator": passport.recommended_operator},
            ),
        )
        self._complete_step(
            job_id,
            "mesh_passport",
            {
                "report": str(report_path),
                "risk_level": passport.risk_level,
                "recommended_operator": passport.recommended_operator,
                "mesh_head_ready": str(passport.mesh_head_ready).lower(),
            },
        )

    def _run_reference_refinement(self, job_id: str, proxy_mesh: Path | None) -> Path | None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "reference_refinement", JobStepStatus.RUNNING)
        if proxy_mesh is None:
            self._skip_step(job_id, "reference_refinement", "No proxy mesh available")
            return None
        if not self.execute_heavy:
            self._skip_step(job_id, "reference_refinement", "execute_heavy is false", {"mesh": str(proxy_mesh)})
            return None
        if not self._metadata_bool(job_id, "reference_refinement_enabled", default=True):
            self._skip_step(job_id, "reference_refinement", "reference_refinement_enabled is false", {"mesh": str(proxy_mesh)})
            return None

        supplied = (
            job.request.metadata.get("reference_mesh_path")
            or job.request.metadata.get("refined_reference_mesh_path")
            or job.request.metadata.get("ultrashape_refined_mesh_path")
            or job.request.metadata.get("manifold_mesh_path")
        )
        if supplied:
            refined_path = Path(str(supplied)).expanduser()
            if not refined_path.exists():
                raise FileNotFoundError(f"reference refinement mesh does not exist: {refined_path}")
            refined_path, filter_artifacts = self._filter_reference_components(job_id, refined_path)
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="refined_reference_mesh",
                    uri=str(refined_path),
                    content_type="model/mesh",
                    metadata={"source": str(proxy_mesh), "stage": "metadata_supplied", **filter_artifacts},
                ),
            )
            self._complete_step(job_id, "reference_refinement", {"mesh": str(refined_path), "source_mesh": str(proxy_mesh), "engine": "metadata", **filter_artifacts})
            return refined_path

        command = (
            job.request.metadata.get("reference_refinement_command")
            or job.request.metadata.get("ultrashape_command")
            or job.request.metadata.get("manifoldization_command")
        )
        if command:
            output_dir = self.artifacts.path(project_id, job_id, "reference_refinement", "")
            output_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = root / "logs" / "reference_refinement.stdout.log"
            stderr_path = root / "logs" / "reference_refinement.stderr.log"
            before = time.time()
            rendered = run_logged_command(
                command,
                cwd=job.request.metadata.get("reference_refinement_cwd") or job.request.metadata.get("ultrashape_cwd"),
                values=self._command_values(
                    job,
                    root,
                    extra={
                        "input_mesh": str(proxy_mesh),
                        "proxy_mesh": str(proxy_mesh),
                        "reference_image": self._step_artifact(job, "input_validation", "input_path") or job.request.input_uri,
                        "output_dir": str(output_dir),
                    },
                ),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                env=job.request.metadata.get("reference_refinement_env") or job.request.metadata.get("ultrashape_env"),
                timeout_seconds=int(job.request.metadata.get("reference_refinement_timeout_seconds", job.request.metadata.get("ultrashape_timeout_seconds", 7200))),
            )
            refined_path = discover_new_mesh(output_dir, since_mtime=before) or discover_new_mesh(root, since_mtime=before)
            if refined_path is None:
                raise RuntimeError(f"reference refinement completed but no mesh was found under {output_dir}")
            refined_path, filter_artifacts = self._filter_reference_components(job_id, refined_path)
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="refined_reference_mesh",
                    uri=str(refined_path),
                    content_type="model/mesh",
                    metadata={"source": str(proxy_mesh), "command": rendered, "stdout": str(stdout_path), "stderr": str(stderr_path), **filter_artifacts},
                ),
            )
            self._complete_step(
                job_id,
                "reference_refinement",
                {"mesh": str(refined_path), "source_mesh": str(proxy_mesh), "stdout": str(stdout_path), "stderr": str(stderr_path), "engine": "command", **filter_artifacts},
            )
            return refined_path

        local_engine = str(job.request.metadata.get("reference_refinement_engine", "")).lower()
        if local_engine in {"poisson", "cleanup", "auto"}:
            from dataclasses import asdict

            from clearmesh.mesh.normalization import normalize_surface_file, normalization_options_from_metadata

            output_dir = self.artifacts.path(project_id, job_id, "reference_refinement", "")
            suffix = Path(proxy_mesh).suffix or ".obj"
            refined_path = output_dir / f"{Path(proxy_mesh).stem}_reference{suffix}"
            report_path = root / "reports" / "reference_refinement.json"
            options = normalization_options_from_metadata({**job.request.metadata, "surface_engine": local_engine})
            report = normalize_surface_file(proxy_mesh, refined_path, options)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
            refined_path, filter_artifacts = self._filter_reference_components(job_id, refined_path)
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="reference_mesh",
                    uri=str(refined_path),
                    content_type="model/mesh",
                    metadata={"source": str(proxy_mesh), "engine": local_engine, "report": str(report_path), **filter_artifacts},
                ),
            )
            self._complete_step(
                job_id,
                "reference_refinement",
                {"mesh": str(refined_path), "source_mesh": str(proxy_mesh), "engine": local_engine, "report": str(report_path), **filter_artifacts},
            )
            return refined_path

        report = root / "reports" / "reference_refinement.todo.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "Supply metadata.reference_refinement_command, metadata.ultrashape_command, "
            "metadata.manifoldization_command, metadata.reference_mesh_path, or "
            "metadata.reference_refinement_engine=poisson|cleanup|auto.\n",
            encoding="utf-8",
        )
        self._skip_step(job_id, "reference_refinement", "UltraShape/manifoldization hook not configured", {"mesh": str(proxy_mesh), "report": str(report)})
        return None

    def _filter_reference_components(self, job_id: str, reference_mesh: Path) -> tuple[Path, dict[str, str]]:
        job, project_id, root = self._job_dirs(job_id)
        if not self._metadata_bool(job_id, "reference_component_filter_enabled", default=True):
            return reference_mesh, {}

        from dataclasses import asdict

        from clearmesh.mesh.cleanup import CleanupOptions, cleanup_mesh_file

        threshold = float(job.request.metadata.get("reference_dominant_component_face_ratio", 0.9))
        output_dir = self.artifacts.path(project_id, job_id, "reference_refinement", "")
        filtered_path = output_dir / f"{Path(reference_mesh).stem}_dominant{Path(reference_mesh).suffix or '.glb'}"
        report_path = root / "reports" / "reference_component_filter.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = cleanup_mesh_file(
            reference_mesh,
            filtered_path,
            CleanupOptions(
                min_component_faces=1,
                min_component_face_ratio=0.0,
                dominant_component_face_ratio=threshold,
                fill_holes=True,
                fix_normals=True,
                merge_vertices=True,
            ),
        )
        report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
        if report.output_faces == report.input_faces and report.output_components == report.input_components:
            try:
                filtered_path.unlink(missing_ok=True)
            except OSError:
                pass
            return reference_mesh, {"component_filter_report": str(report_path), "component_filter_applied": "false"}
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="refined_reference_mesh",
                uri=str(filtered_path),
                content_type="model/mesh",
                metadata={"source": str(reference_mesh), "stage": "dominant_component_filter", "report": str(report_path)},
            ),
        )
        return filtered_path, {"component_filter_report": str(report_path), "component_filter_applied": "true", "unfiltered_mesh": str(reference_mesh)}

    def _run_surface_normalization(self, job_id: str, proxy_mesh: Path | None) -> Path | None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "surface_normalization", JobStepStatus.RUNNING)
        if proxy_mesh is None:
            self._skip_step(job_id, "surface_normalization", "No proxy mesh available")
            return None
        if not self.execute_heavy:
            self._skip_step(job_id, "surface_normalization", "execute_heavy is false", {"mesh": str(proxy_mesh)})
            return proxy_mesh
        if str(job.request.metadata.get("surface_normalization_enabled", "true")).lower() in {"0", "false", "no", "off"}:
            self._skip_step(job_id, "surface_normalization", "surface_normalization_enabled is false", {"mesh": str(proxy_mesh)})
            return proxy_mesh

        from dataclasses import asdict

        from clearmesh.mesh.normalization import normalize_surface_file, normalization_options_from_metadata

        output_dir = self.artifacts.path(project_id, job_id, "surface", "")
        suffix = Path(proxy_mesh).suffix or ".obj"
        control_mesh = output_dir / f"{Path(proxy_mesh).stem}_control{suffix}"
        report_path = root / "reports" / "surface_normalization.json"
        report = normalize_surface_file(
            proxy_mesh,
            control_mesh,
            normalization_options_from_metadata(job.request.metadata),
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="control_mesh",
                uri=str(control_mesh),
                content_type="model/mesh",
                metadata={"source": str(proxy_mesh), "report": str(report_path), "engine": report.engine},
            ),
        )
        self._complete_step(
            job_id,
            "surface_normalization",
            {"mesh": str(control_mesh), "report": str(report_path), "engine": report.engine, "source_mesh": str(proxy_mesh)},
        )
        return control_mesh

    def _run_shrinkwrap_projection(self, job_id: str, control_mesh: Path | None, target_mesh: Path | None) -> Path | None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "shrinkwrap_projection", JobStepStatus.RUNNING)
        if control_mesh is None or target_mesh is None:
            self._skip_step(job_id, "shrinkwrap_projection", "No control or target mesh available")
            return control_mesh
        if not self.execute_heavy:
            self._skip_step(job_id, "shrinkwrap_projection", "execute_heavy is false", {"mesh": str(control_mesh)})
            return control_mesh
        if str(job.request.metadata.get("shrinkwrap_enabled", "false")).lower() not in {"1", "true", "yes", "on"}:
            self._skip_step(job_id, "shrinkwrap_projection", "shrinkwrap_enabled is false", {"mesh": str(control_mesh)})
            return control_mesh

        from dataclasses import asdict

        from clearmesh.mesh.shrinkwrap import shrinkwrap_file, shrinkwrap_obj_vertices_file, shrinkwrap_options_from_metadata

        output_dir = self.artifacts.path(project_id, job_id, "shrinkwrap", "")
        suffix = Path(control_mesh).suffix or ".obj"
        wrapped_path = output_dir / f"{Path(control_mesh).stem}_projected{suffix}"
        report_path = root / "reports" / "shrinkwrap_projection.json"
        report = shrinkwrap_file(
            control_mesh,
            target_mesh,
            wrapped_path,
            shrinkwrap_options_from_metadata(job.request.metadata),
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="control_mesh",
                uri=str(wrapped_path),
                content_type="model/mesh",
                metadata={"source": str(control_mesh), "target": str(target_mesh), "report": str(report_path), "stage": "shrinkwrap"},
            ),
        )
        self._complete_step(
            job_id,
            "shrinkwrap_projection",
            {"mesh": str(wrapped_path), "report": str(report_path), "source_mesh": str(control_mesh), "target_mesh": str(target_mesh)},
        )
        return wrapped_path

    def _run_retopology_planning(self, job_id: str, control_mesh: Path | None) -> None:
        job, _, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "retopology_planning", JobStepStatus.RUNNING)
        report_path = root / "reports" / "retopology_plan.json"
        if control_mesh is None:
            self._skip_step(job_id, "retopology_planning", "No control mesh available")
            return
        if not self.execute_heavy:
            self._skip_step(job_id, "retopology_planning", "execute_heavy is false", {"mesh": str(control_mesh)})
            return
        if not self._metadata_bool(job_id, "retopology_planning_enabled", default=True):
            self._skip_step(job_id, "retopology_planning", "retopology_planning_enabled is false", {"mesh": str(control_mesh)})
            return

        from clearmesh.retopology.feature_graph import (
            analyze_retopology_file,
            retopology_planning_options_from_metadata,
            write_retopology_plan,
        )

        plan = analyze_retopology_file(control_mesh, retopology_planning_options_from_metadata(job.request.metadata))
        write_retopology_plan(report_path, plan)
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="retopology_plan",
                uri=str(report_path),
                content_type="application/json",
                metadata={
                    "source": str(control_mesh),
                    "risk_level": str(plan.summary.get("risk_level", "")),
                    "chart_count": str(plan.summary.get("chart_count", "")),
                    "feature_curve_count": str(plan.summary.get("feature_curve_count", "")),
                },
            ),
        )
        self._complete_step(
            job_id,
            "retopology_planning",
            {
                "report": str(report_path),
                "mesh": str(control_mesh),
                "risk_level": str(plan.summary.get("risk_level", "")),
                "chart_count": str(plan.summary.get("chart_count", "")),
                "feature_curve_count": str(plan.summary.get("feature_curve_count", "")),
            },
        )

    def _run_chart_remesh(self, job_id: str, control_mesh: Path | None) -> None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "chart_remesh", JobStepStatus.RUNNING)
        if control_mesh is None:
            self._skip_step(job_id, "chart_remesh", "No control mesh available")
            return
        if not self.execute_heavy:
            self._skip_step(job_id, "chart_remesh", "execute_heavy is false", {"mesh": str(control_mesh)})
            return
        if not self._metadata_bool(job_id, "chart_remesh_enabled", default=False):
            self._skip_step(job_id, "chart_remesh", "chart_remesh_enabled is false", {"mesh": str(control_mesh)})
            return
        plan_path = self._step_artifact(job, "retopology_planning", "report")
        if not plan_path:
            self._skip_step(job_id, "chart_remesh", "No retopology plan available", {"mesh": str(control_mesh)})
            return

        from clearmesh.retopology.chart_remesh import (
            chart_remesh_options_from_metadata,
            remesh_plan_charts,
            write_chart_remesh_manifest,
        )

        output_dir = self.artifacts.path(project_id, job_id, "chart_remesh", "")
        manifest_path = root / "reports" / "chart_remesh_manifest.json"
        manifest = remesh_plan_charts(control_mesh, plan_path, output_dir, chart_remesh_options_from_metadata(job.request.metadata))
        write_chart_remesh_manifest(manifest_path, manifest)
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="chart_remesh_manifest",
                uri=str(manifest_path),
                content_type="application/json",
                metadata={
                    "source": str(control_mesh),
                    "plan": str(plan_path),
                    "succeeded_chart_count": str(manifest.summary.get("succeeded_chart_count", "")),
                    "attempted_chart_count": str(manifest.summary.get("attempted_chart_count", "")),
                },
            ),
        )
        self._complete_step(
            job_id,
            "chart_remesh",
            {
                "manifest": str(manifest_path),
                "output_dir": str(output_dir),
                "mesh": str(control_mesh),
                "succeeded_chart_count": str(manifest.summary.get("succeeded_chart_count", "")),
            },
        )

    def _run_chart_stitch(self, job_id: str) -> Path | None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "chart_stitch", JobStepStatus.RUNNING)
        if not self.execute_heavy:
            self._skip_step(job_id, "chart_stitch", "execute_heavy is false")
            return None
        if not self._metadata_bool(job_id, "chart_stitch_enabled", default=False):
            self._skip_step(job_id, "chart_stitch", "chart_stitch_enabled is false")
            return None
        manifest_path = self._step_artifact(job, "chart_remesh", "manifest")
        if not manifest_path:
            self._skip_step(job_id, "chart_stitch", "No chart remesh manifest available")
            return None

        from clearmesh.retopology.chart_stitch import (
            chart_stitch_options_from_metadata,
            stitch_chart_remesh_outputs,
            write_chart_stitch_report,
        )

        output_dir = self.artifacts.path(project_id, job_id, "chart_stitch", "")
        stitched_path = output_dir / "stitched_chart_quads.obj"
        report_path = root / "reports" / "chart_stitch.json"
        report = stitch_chart_remesh_outputs(manifest_path, stitched_path, chart_stitch_options_from_metadata(job.request.metadata))
        write_chart_stitch_report(report_path, report)
        promoted = bool(report.promotion.get("promoted"))
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="chart_stitched_mesh",
                uri=str(stitched_path),
                content_type="model/obj",
                metadata={
                    "manifest": str(manifest_path),
                    "report": str(report_path),
                    "promoted": str(promoted).lower(),
                    "quad_ratio": str(report.quad_stats.get("quad_ratio", "")),
                    "stitched_chart_count": str(report.stitched_chart_count),
                    "as_final": str(self._metadata_bool(job_id, "chart_stitch_as_final", default=False) and promoted).lower(),
                },
            ),
        )
        self._complete_step(
            job_id,
            "chart_stitch",
            {
                "mesh": str(stitched_path),
                "report": str(report_path),
                "manifest": str(manifest_path),
                "promoted": str(promoted).lower(),
                "quad_ratio": str(report.quad_stats.get("quad_ratio", "")),
                "stitched_chart_count": str(report.stitched_chart_count),
            },
        )
        return stitched_path if promoted or self._metadata_bool(job_id, "chart_stitch_allow_unpromoted_projection", default=False) else None

    def _run_quad_remesh(self, job_id: str, control_mesh: Path | None) -> Path | None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "quad_remesh", JobStepStatus.RUNNING)
        if control_mesh is None:
            self._skip_step(job_id, "quad_remesh", "No control mesh available")
            return None
        if not self.execute_heavy:
            self._skip_step(job_id, "quad_remesh", "execute_heavy is false", {"mesh": str(control_mesh)})
            return None
        if not self._metadata_bool(job_id, "quad_remesh_enabled", default=False):
            self._skip_step(job_id, "quad_remesh", "quad_remesh_enabled is false", {"mesh": str(control_mesh)})
            return None

        from dataclasses import asdict

        from clearmesh.retopology.quad_remesh import quad_remesh_file, quad_remesh_options_from_metadata

        output_dir = self.artifacts.path(project_id, job_id, "quad_remesh", "")
        quad_path = output_dir / f"{Path(control_mesh).stem}_quad.obj"
        report_path = root / "reports" / "quad_remesh.json"
        report = quad_remesh_file(control_mesh, quad_path, quad_remesh_options_from_metadata(job.request.metadata))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
        quad_ratio = str(report.quad_stats.get("quad_ratio", ""))
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="quad_mesh",
                uri=str(quad_path),
                content_type="model/obj",
                metadata={
                    "source": str(control_mesh),
                    "report": str(report_path),
                    "engine": report.engine,
                    "quad_ratio": quad_ratio,
                    "as_final": str(self._metadata_bool(job_id, "quad_remesh_as_final", default=False)).lower(),
                },
            ),
        )
        self._complete_step(
            job_id,
            "quad_remesh",
            {"mesh": str(quad_path), "report": str(report_path), "engine": report.engine, "quad_ratio": quad_ratio, "source_mesh": str(control_mesh)},
        )
        return quad_path

    def _run_feature_projection(self, job_id: str, quad_mesh: Path | None, target_mesh: Path | None) -> Path | None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "feature_projection", JobStepStatus.RUNNING)
        if quad_mesh is None:
            self._skip_step(job_id, "feature_projection", "No quad mesh available")
            return None
        if target_mesh is None:
            self._skip_step(job_id, "feature_projection", "No reference target mesh available", {"mesh": str(quad_mesh)})
            return None
        if not self.execute_heavy:
            self._skip_step(job_id, "feature_projection", "execute_heavy is false", {"mesh": str(quad_mesh), "target_mesh": str(target_mesh)})
            return None
        if not self._metadata_bool(job_id, "feature_projection_enabled", default=False):
            self._skip_step(job_id, "feature_projection", "feature_projection_enabled is false", {"mesh": str(quad_mesh), "target_mesh": str(target_mesh)})
            return None

        from dataclasses import asdict

        from clearmesh.mesh.shrinkwrap import shrinkwrap_file, shrinkwrap_obj_vertices_file, shrinkwrap_options_from_metadata

        output_dir = self.artifacts.path(project_id, job_id, "feature_projection", "")
        suffix = Path(quad_mesh).suffix or ".obj"
        projected_path = output_dir / f"{Path(quad_mesh).stem}_projected{suffix}"
        report_path = root / "reports" / "feature_projection.json"
        projector = shrinkwrap_obj_vertices_file if Path(quad_mesh).suffix.lower() == ".obj" else shrinkwrap_file
        report = projector(
            quad_mesh,
            target_mesh,
            projected_path,
            shrinkwrap_options_from_metadata({**job.request.metadata, "shrinkwrap_enabled": True}),
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="projected_quad_mesh",
                uri=str(projected_path),
                content_type="model/mesh",
                metadata={
                    "source": str(quad_mesh),
                    "target": str(target_mesh),
                    "report": str(report_path),
                    "as_final": str(self._metadata_bool(job_id, "feature_projection_as_final", default=False)).lower(),
                },
            ),
        )
        self._complete_step(
            job_id,
            "feature_projection",
            {"mesh": str(projected_path), "report": str(report_path), "source_mesh": str(quad_mesh), "target_mesh": str(target_mesh)},
        )
        return projected_path

    def _run_preview_publish(self, job_id: str, mesh_path: Path | None) -> None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "preview_publish", JobStepStatus.RUNNING)
        report_path = root / "reports" / "preview_publish.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if mesh_path is None:
            report_path.write_text("No mesh available for preview.\n", encoding="utf-8")
            self._skip_step(job_id, "preview_publish", "No mesh available", {"report": str(report_path)})
            return
        preview_dir = self.artifacts.path(project_id, job_id, "previews", "")
        preview_path = preview_dir / "control_preview.png"
        artifacts = {"mesh": str(mesh_path), "report": str(report_path)}
        try:
            from clearmesh.mesh.preview import render_wire_preview_file

            render_wire_preview_file(mesh_path, preview_path, size=int(job.request.metadata.get("preview_size", 1024)))
            asset_uri = str(preview_path)
            upload_file = getattr(self.artifacts, "upload_file", None)
            if callable(upload_file):
                asset_uri = upload_file(preview_path, project_id, job_id, "previews", preview_path.name)
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="preview_image",
                    uri=asset_uri,
                    content_type="image/png",
                    metadata={"stage": "control_surface", "source_mesh": str(mesh_path)},
                ),
            )
            artifacts["preview_image"] = str(preview_path)
            report_path.write_text(f"preview_image={preview_path}\nsource_mesh={mesh_path}\n", encoding="utf-8")
            self._complete_step(job_id, "preview_publish", artifacts)
        except Exception as exc:  # noqa: BLE001 - preview should not block mesh delivery.
            report_path.write_text(
                f"Preview render skipped: {type(exc).__name__}: {exc}\nsource_mesh={mesh_path}\n",
                encoding="utf-8",
            )
            self._skip_step(job_id, "preview_publish", "Preview render failed; continuing", artifacts)

    def _run_point_cloud_bridge(self, job_id: str, proxy_mesh: Path | None) -> Path | None:
        job, project_id, _ = self._job_dirs(job_id)
        self.store.update_step(job_id, "point_cloud_bridge", JobStepStatus.RUNNING)
        if proxy_mesh is None:
            self._skip_step(job_id, "point_cloud_bridge", "No proxy mesh available")
            return None
        if not self.execute_heavy:
            self._skip_step(job_id, "point_cloud_bridge", "execute_heavy is false", {"proxy_mesh": str(proxy_mesh)})
            return None

        from clearmesh.pointcloud import sample_to_files

        budget = self.preferred_point_budget
        if job.request.point_budgets:
            budget = min(job.request.point_budgets, key=lambda value: abs(value - self.preferred_point_budget))
        output_prefix = self.artifacts.path(project_id, job_id, "pointclouds", f"{job.id}_{budget}")
        paths = sample_to_files(proxy_mesh, output_prefix, count=budget)
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="point_cloud",
                uri=str(paths["ply"]),
                content_type="model/ply",
                metadata={"sample_count": budget, "npz": str(paths["npz"])},
            ),
        )
        self._complete_step(job_id, "point_cloud_bridge", {"ply": str(paths["ply"]), "npz": str(paths["npz"])})
        return paths["ply"]

    def _run_part_structure(self, job_id: str) -> None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "part_structure", JobStepStatus.RUNNING)
        if not job.request.enable_parts:
            self._skip_step(job_id, "part_structure", "enable_parts is false")
            return
        command = job.request.metadata.get("part_structure_command")
        if command and self.execute_heavy:
            output_dir = root / "parts"
            output_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = root / "logs" / "part_structure.stdout.log"
            stderr_path = root / "logs" / "part_structure.stderr.log"
            rendered = run_logged_command(
                command,
                cwd=job.request.metadata.get("part_structure_cwd"),
                values=self._command_values(
                    job,
                    root,
                    extra={
                        "output_dir": str(output_dir),
                        "proxy_mesh": self._step_artifact(job, "shrinkwrap_projection", "mesh")
                        or self._step_artifact(job, "surface_normalization", "mesh")
                        or self._step_artifact(job, "trellis_proxy", "proxy_mesh")
                        or "",
                        "point_cloud": self._step_artifact(job, "point_cloud_bridge", "ply") or "",
                        "part_mask_path": self._resolve_metadata_local_uri(job, "part_mask_uri") or str(job.request.metadata.get("part_mask_path", "")),
                    },
                ),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                env=job.request.metadata.get("part_structure_env"),
                timeout_seconds=int(job.request.metadata.get("part_structure_timeout_seconds", 3600)),
            )
            manifest = output_dir / "parts_manifest.json"
            artifacts = {"output_dir": str(output_dir), "stdout": str(stdout_path), "stderr": str(stderr_path)}
            if manifest.exists():
                artifacts["manifest"] = str(manifest)
                self.store.add_asset(
                    job_id,
                    AssetRecord(
                        id=f"asset_{uuid4().hex}",
                        job_id=job_id,
                        kind="part_manifest",
                        uri=str(manifest),
                        content_type="application/json",
                        metadata={"command": rendered},
                    ),
                )
            self._complete_step(job_id, "part_structure", artifacts)
            return
        fallback = str(job.request.metadata.get("part_structure_fallback", "")).lower()
        if fallback in {"components", "connected_components", "meshmosaic_components"} or self._metadata_bool(job_id, "meshmosaic_component_fallback_enabled", default=False):
            from clearmesh.parts.component_manifest import write_component_parts_manifest

            source_mesh = (
                self._step_artifact(job, "feature_projection", "mesh")
                or self._step_artifact(job, "shrinkwrap_projection", "mesh")
                or self._step_artifact(job, "surface_normalization", "mesh")
                or self._step_artifact(job, "reference_refinement", "mesh")
                or self._step_artifact(job, "trellis_proxy", "proxy_mesh")
            )
            if source_mesh:
                output_dir = self.artifacts.path(project_id, job_id, "parts", "")
                manifest = write_component_parts_manifest(
                    source_mesh,
                    output_dir,
                    max_parts=int(job.request.metadata.get("component_part_max_parts", 8)),
                    min_faces=int(job.request.metadata.get("component_part_min_faces", 32)),
                    point_count=int(job.request.metadata.get("component_part_point_count", 4096)),
                )
                self.store.add_asset(
                    job_id,
                    AssetRecord(
                        id=f"asset_{uuid4().hex}",
                        job_id=job_id,
                        kind="part_manifest",
                        uri=str(manifest),
                        content_type="application/json",
                        metadata={"source": source_mesh, "adapter": "connected_components"},
                    ),
                )
                self._complete_step(job_id, "part_structure", {"manifest": str(manifest), "output_dir": str(output_dir), "source_mesh": source_mesh})
                return
        report = root / "reports" / "part_structure.todo.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("Supply metadata.part_structure_command for OmniPart-style execution. Whole-object mesh head continues without it.\n", encoding="utf-8")
        self._skip_step(job_id, "part_structure", "OmniPart-style adapter pending", {"report": str(report)})

    def _run_easy3e_edit(self, job_id: str) -> Path | None:
        job, _, root = self._job_dirs(job_id)
        if not any(step.name == "easy3e_edit" for step in job.steps):
            return None
        self.store.update_step(job_id, "easy3e_edit", JobStepStatus.RUNNING)

        edited_mesh = job.request.metadata.get("edited_mesh_path")
        if edited_mesh:
            edited_mesh_path = Path(str(edited_mesh)).expanduser()
            if not edited_mesh_path.exists():
                raise FileNotFoundError(f"edited_mesh_path does not exist: {edited_mesh_path}")
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="edited_mesh",
                    uri=str(edited_mesh_path),
                    content_type="model/mesh",
                    metadata={"source": "metadata.edited_mesh_path", "editor": "easy3e"},
                ),
            )
            self._complete_step(job_id, "easy3e_edit", {"mesh": str(edited_mesh_path)})
            return edited_mesh_path

        command = job.request.metadata.get("easy3e_command")
        if command and self.execute_heavy:
            output_dir = root / "easy3e"
            output_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = root / "logs" / "easy3e.stdout.log"
            stderr_path = root / "logs" / "easy3e.stderr.log"
            before = time.time()
            rendered = run_logged_command(
                command,
                cwd=job.request.metadata.get("easy3e_cwd"),
                values=self._command_values(job, root, extra={"output_dir": str(output_dir)}),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                env=job.request.metadata.get("easy3e_env"),
                timeout_seconds=int(job.request.metadata.get("easy3e_timeout_seconds", 7200)),
            )
            mesh_path = discover_new_mesh(output_dir, since_mtime=before) or discover_new_mesh(root, since_mtime=before)
            if mesh_path is None:
                raise RuntimeError(f"Easy3E command completed but no mesh was found under {output_dir}")
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="edited_mesh",
                    uri=str(mesh_path),
                    content_type="model/mesh",
                    metadata={"command": rendered, "stdout": str(stdout_path), "stderr": str(stderr_path), "editor": "easy3e"},
                ),
            )
            self._complete_step(job_id, "easy3e_edit", {"mesh": str(mesh_path), "stdout": str(stdout_path), "stderr": str(stderr_path)})
            return mesh_path

        report = root / "reports" / "easy3e_edit.todo.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "Supply metadata.edited_mesh_path or metadata.easy3e_command for edit_image/edit_text jobs.\n",
            encoding="utf-8",
        )
        self._skip_step(job_id, "easy3e_edit", "Easy3E adapter pending", {"report": str(report)})
        return None

    def _run_mesh_head(self, job_id: str, point_cloud: Path | None, proxy_mesh: Path | None) -> Path | None:
        job, project_id, _ = self._job_dirs(job_id)
        self.store.update_step(job_id, "mesh_head", JobStepStatus.RUNNING)
        artist_mesh = job.request.metadata.get("artist_mesh_path")
        if artist_mesh:
            artist_mesh_path = Path(str(artist_mesh)).expanduser()
            if not artist_mesh_path.exists():
                raise FileNotFoundError(f"artist_mesh_path does not exist: {artist_mesh_path}")
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="artist_mesh",
                    uri=str(artist_mesh_path),
                    content_type="model/mesh",
                    metadata={"source": "metadata.artist_mesh_path"},
                ),
            )
            self._complete_step(job_id, "mesh_head", {"mesh": str(artist_mesh_path)})
            return artist_mesh_path
        if point_cloud is None:
            self._skip_step(job_id, "mesh_head", "No point cloud available")
            return None
        if not self.execute_heavy:
            self._skip_step(job_id, "mesh_head", "execute_heavy is false", {"point_cloud": str(point_cloud)})
            return None
        policy = str(job.request.metadata.get("mesh_head_policy", "passport")).lower()
        if policy in {"0", "false", "no", "off", "never"}:
            self._skip_step(job_id, "mesh_head", "mesh_head_policy disables learned refinement", {"point_cloud": str(point_cloud)})
            return None
        if policy not in {"always", "force"} and self._step_artifact(job, "mesh_passport", "mesh_head_ready") != "true":
            self._skip_step(
                job_id,
                "mesh_head",
                "mesh passport did not approve expensive learned refinement",
                {"point_cloud": str(point_cloud), "policy": policy},
            )
            return None

        part_meshes = self._run_part_mesh_heads(job, project_id, point_cloud, proxy_mesh)
        if part_meshes:
            manifest = self.artifacts.path(project_id, job_id, "mesh_heads", self.mesh_head, "part_mesh_manifest.json")
            manifest.write_text(json.dumps({"job_id": job.id, "mesh_head": self.mesh_head, "parts": part_meshes}, indent=2, sort_keys=True), encoding="utf-8")
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="part_mesh_manifest",
                    uri=str(manifest),
                    content_type="application/json",
                    metadata={"mesh_head": self.mesh_head, "part_count": len(part_meshes)},
                ),
            )
            primary_mesh = Path(part_meshes[0]["mesh_path"])
            self._complete_step(job_id, "mesh_head", {"manifest": str(manifest), "mesh": str(primary_mesh), "part_count": str(len(part_meshes))})
            return primary_mesh

        from clearmesh.mesh_heads import MeshHeadInput, build_mesh_head

        output_dir = self.artifacts.path(project_id, job_id, "mesh_heads", self.mesh_head)
        adapter = build_mesh_head(self.mesh_head, self.mesh_head_config)
        result = adapter.run(
            MeshHeadInput(
                case_id=job.id,
                point_cloud_path=point_cloud,
                proxy_mesh_path=proxy_mesh,
                output_dir=output_dir,
                metadata={"mode": job.request.mode, "quality_tier": job.request.quality_tier},
            )
        )
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="artist_mesh",
                uri=str(result.mesh_path),
                content_type="model/mesh",
                metadata={"mesh_head": result.adapter_name, "stdout": str(result.stdout_path), "stderr": str(result.stderr_path)},
            ),
        )
        self._complete_step(job_id, "mesh_head", {"mesh": str(result.mesh_path), "stdout": str(result.stdout_path), "stderr": str(result.stderr_path)})
        return result.mesh_path

    def _run_part_mesh_heads(self, job: JobRecord, project_id: str, point_cloud: Path, proxy_mesh: Path | None) -> list[dict[str, str]]:
        if str(job.request.metadata.get("part_mesh_generation", "false")).lower() not in {"1", "true", "yes", "on"}:
            return []
        manifest_path = self._step_artifact(job, "part_structure", "manifest")
        if not manifest_path:
            return []

        from clearmesh.mesh_heads import MeshHeadInput, build_mesh_head
        from clearmesh.parts.manifest import load_parts_manifest

        parts = load_parts_manifest(manifest_path)
        if not parts:
            return []
        max_parts = int(job.request.metadata.get("part_mesh_max_parts", 8))
        adapter = build_mesh_head(self.mesh_head, self.mesh_head_config)
        outputs: list[dict[str, str]] = []
        for part in parts[:max_parts]:
            part_cloud = Path(part.point_cloud_path).expanduser() if part.point_cloud_path else point_cloud
            part_proxy = Path(part.proxy_mesh_path).expanduser() if part.proxy_mesh_path else proxy_mesh
            output_dir = self.artifacts.path(project_id, job.id, "mesh_heads", self.mesh_head, "parts", part.id)
            result = adapter.run(
                MeshHeadInput(
                    case_id=f"{job.id}_{part.id}",
                    point_cloud_path=part_cloud,
                    proxy_mesh_path=part_proxy,
                    output_dir=output_dir,
                    metadata={
                        "mode": job.request.mode,
                        "quality_tier": job.request.quality_tier,
                        "part_id": part.id,
                        "part_label": part.label or "",
                    },
                )
            )
            self.store.add_asset(
                job.id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job.id,
                    kind="part_artist_mesh",
                    uri=str(result.mesh_path),
                    content_type="model/mesh",
                    metadata={
                        "mesh_head": result.adapter_name,
                        "part_id": part.id,
                        "part_label": part.label or "",
                        "stdout": str(result.stdout_path),
                        "stderr": str(result.stderr_path),
                    },
                ),
            )
            outputs.append(
                {
                    "part_id": part.id,
                    "part_label": part.label or "",
                    "mesh_path": str(result.mesh_path),
                    "stdout": str(result.stdout_path),
                    "stderr": str(result.stderr_path),
                }
            )
        return outputs

    def _run_repair_validation(self, job_id: str, mesh_path: Path | None) -> None:
        job, _, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "repair_validation", JobStepStatus.RUNNING)
        report = root / "reports" / "repair_validation.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        artifacts = {"report": str(report)}
        if mesh_path and self.execute_heavy:
            from clearmesh.eval.mesh_quality import evaluate_mesh, evaluate_mesh_pair

            eval_report: dict[str, object] = {"mesh_path": str(mesh_path), "mesh_metrics": evaluate_mesh(mesh_path)}
            proxy_path = self._step_artifact(job, "surface_normalization", "source_mesh") or self._step_artifact(job, "trellis_proxy", "proxy_mesh")
            if proxy_path:
                try:
                    eval_report["reference_path"] = proxy_path
                    eval_report["pair_metrics"] = evaluate_mesh_pair(mesh_path, proxy_path, samples=10_000)
                except Exception as exc:  # noqa: BLE001 - eval should report, not fail export.
                    eval_report["pair_metrics"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            eval_path = root / "reports" / "mesh_eval.json"
            eval_path.write_text(json.dumps(eval_report, indent=2, sort_keys=True), encoding="utf-8")
            report.write_text(f"mesh_path={mesh_path}\neval_report={eval_path}\n", encoding="utf-8")
            artifacts["eval_report"] = str(eval_path)
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="mesh_eval_report",
                    uri=str(eval_path),
                    content_type="application/json",
                ),
            )
        else:
            report.write_text(f"mesh_path={mesh_path or ''}\nEvaluation skipped because no mesh is available or execute_heavy is false.\n", encoding="utf-8")
        self._complete_step(job_id, "repair_validation", artifacts)

    def _run_production_gate(self, job_id: str, mesh_path: Path | None) -> None:
        job, _, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "production_gate", JobStepStatus.RUNNING)
        report_path = root / "reports" / "production_gate.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if mesh_path is None:
            self._skip_step(job_id, "production_gate", "No mesh available")
            return
        if not self.execute_heavy:
            self._skip_step(job_id, "production_gate", "execute_heavy is false", {"mesh": str(mesh_path)})
            return

        from clearmesh.eval.mesh_quality import evaluate_mesh
        from clearmesh.retopology.quad_remesh import quad_mesh_stats

        mesh_metrics = evaluate_mesh(mesh_path)
        quad_stats = quad_mesh_stats(mesh_path) if Path(mesh_path).suffix.lower() == ".obj" else {}
        blender_report: dict[str, object] | None = None
        if self._metadata_bool(job_id, "blender_gates_enabled", default=False):
            from clearmesh.eval.blender_gates import BlenderGateOptions, run_blender_mesh_gates

            blender_output_dir = root / "reports" / "blender_gates"
            blender_result = run_blender_mesh_gates(
                mesh_path,
                blender_output_dir,
                BlenderGateOptions(
                    blender=str(job.request.metadata.get("blender_path", job.request.metadata.get("blender", "blender"))),
                    timeout_seconds=int(job.request.metadata.get("blender_gate_timeout_seconds", 180)),
                ),
            )
            blender_report = blender_result.to_dict()

        gate = _production_gate_decision(job.request.metadata, mesh_metrics, quad_stats, blender_report)
        payload = {
            "mesh_path": str(mesh_path),
            "mesh_metrics": mesh_metrics,
            "quad_stats": quad_stats,
            "blender_gates": blender_report,
            "promotion": gate,
        }
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="production_gate_report",
                uri=str(report_path),
                content_type="application/json",
                metadata={
                    "source": str(mesh_path),
                    "tier": str(gate.get("tier", "")),
                    "promoted": str(gate.get("promoted", False)).lower(),
                    "editable_quad_ready": str(gate.get("editable_quad_ready", False)).lower(),
                },
            ),
        )
        self._complete_step(
            job_id,
            "production_gate",
            {
                "report": str(report_path),
                "mesh": str(mesh_path),
                "tier": str(gate.get("tier", "")),
                "promoted": str(gate.get("promoted", False)).lower(),
                "editable_quad_ready": str(gate.get("editable_quad_ready", False)).lower(),
            },
        )

    def _run_mesh_cleanup(self, job_id: str, mesh_path: Path | None) -> Path | None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "mesh_cleanup", JobStepStatus.RUNNING)
        if mesh_path is None:
            self._skip_step(job_id, "mesh_cleanup", "No mesh available")
            return None
        if not self.execute_heavy:
            self._skip_step(job_id, "mesh_cleanup", "execute_heavy is false", {"mesh": str(mesh_path)})
            return mesh_path
        if str(job.request.metadata.get("cleanup_enabled", "true")).lower() in {"0", "false", "no", "off"}:
            self._skip_step(job_id, "mesh_cleanup", "cleanup_enabled is false", {"mesh": str(mesh_path)})
            return mesh_path

        from dataclasses import asdict

        from clearmesh.mesh.cleanup import cleanup_mesh_file, cleanup_options_from_metadata

        output_dir = self.artifacts.path(project_id, job_id, "cleanup", "")
        cleaned_path = output_dir / f"{Path(mesh_path).stem}_cleaned{Path(mesh_path).suffix or '.obj'}"
        report_path = output_dir / "cleanup_report.json"
        cleanup_report = cleanup_mesh_file(
            mesh_path,
            cleaned_path,
            options=cleanup_options_from_metadata(job.request.metadata),
        )
        report_path.write_text(json.dumps(asdict(cleanup_report), indent=2, sort_keys=True), encoding="utf-8")
        self.store.add_asset(
            job_id,
            AssetRecord(
                id=f"asset_{uuid4().hex}",
                job_id=job_id,
                kind="cleaned_mesh",
                uri=str(cleaned_path),
                content_type="model/mesh",
                metadata={"source": str(mesh_path), "report": str(report_path)},
            ),
        )
        self._complete_step(job_id, "mesh_cleanup", {"mesh": str(cleaned_path), "report": str(report_path), "source_mesh": str(mesh_path)})
        return cleaned_path

    def _run_export_package(self, job_id: str, mesh_path: Path | None) -> None:
        job, project_id, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "export_package", JobStepStatus.RUNNING)
        package = root / "exports" / "package_manifest.json"
        package.parent.mkdir(parents=True, exist_ok=True)
        exports: list[dict[str, str]] = []
        if mesh_path and Path(mesh_path).exists():
            copied = self.artifacts.put_file(mesh_path, project_id, job_id, "exports", Path(mesh_path).name)
            exports.append({"kind": "source_mesh", "path": str(copied)})
            asset_uri = str(copied)
            upload_file = getattr(self.artifacts, "upload_file", None)
            if callable(upload_file):
                asset_uri = upload_file(copied, project_id, job_id, "exports", Path(mesh_path).name)
            self.store.add_asset(
                job_id,
                AssetRecord(
                    id=f"asset_{uuid4().hex}",
                    job_id=job_id,
                    kind="export_mesh",
                    uri=asset_uri,
                    content_type="model/mesh",
                    metadata={"source": str(mesh_path)},
                ),
            )
        package.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "status": "ready" if exports else "scaffold",
                    "requested_formats": job.request.output_formats,
                    "exports": exports,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self._complete_step(job_id, "export_package", {"manifest": str(package)})

    def _run_autorigging(self, job_id: str, mesh_path: Path | None) -> None:
        job, _, root = self._job_dirs(job_id)
        self.store.update_step(job_id, "autorigging", JobStepStatus.RUNNING)
        command = job.request.metadata.get("autorigging_command")
        if command and mesh_path and self.execute_heavy:
            output_dir = root / "rigging"
            output_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = root / "logs" / "autorigging.stdout.log"
            stderr_path = root / "logs" / "autorigging.stderr.log"
            rendered = run_logged_command(
                command,
                cwd=job.request.metadata.get("autorigging_cwd"),
                values=self._command_values(job, root, extra={"mesh_path": str(mesh_path), "output_dir": str(output_dir)}),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                env=job.request.metadata.get("autorigging_env"),
                timeout_seconds=int(job.request.metadata.get("autorigging_timeout_seconds", 3600)),
            )
            rig_path = discover_new_mesh(output_dir, since_mtime=0) or discover_new_mesh(root, since_mtime=0)
            artifacts = {"output_dir": str(output_dir), "stdout": str(stdout_path), "stderr": str(stderr_path)}
            if rig_path is not None:
                artifacts["rigged_mesh"] = str(rig_path)
                self.store.add_asset(
                    job_id,
                    AssetRecord(
                        id=f"asset_{uuid4().hex}",
                        job_id=job_id,
                        kind="rigged_mesh",
                        uri=str(rig_path),
                        content_type="model/mesh",
                        metadata={"command": rendered},
                    ),
                )
            self._complete_step(job_id, "autorigging", artifacts)
            return
        report = root / "reports" / "autorigging.todo.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(f"Optional autorigging hook pending. mesh_path={mesh_path or ''}\n", encoding="utf-8")
        self._skip_step(job_id, "autorigging", "Optional autorigging adapter pending", {"report": str(report)})

    def _command_values(self, job: JobRecord, root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
        values = {
            "job_id": job.id,
            "input_uri": job.request.input_uri,
            "input_path": self._step_artifact(job, "input_validation", "input_path") or job.request.input_uri,
            "prompt": job.request.prompt or "",
            "root": str(root),
            "reports_dir": str(root / "reports"),
            "logs_dir": str(root / "logs"),
            "source_mesh_path": str(job.request.metadata.get("source_mesh_path", "")),
            "coarse_proxy_mesh_path": self._step_artifact(job, "coarse_adapter", "mesh") or "",
            "reference_mesh_path": self._step_artifact(job, "reference_refinement", "mesh") or "",
            "control_mesh_path": self._step_artifact(job, "shrinkwrap_projection", "mesh") or self._step_artifact(job, "surface_normalization", "mesh") or "",
            "retopology_plan_path": self._step_artifact(job, "retopology_planning", "report") or "",
            "chart_remesh_manifest_path": self._step_artifact(job, "chart_remesh", "manifest") or "",
            "chart_stitched_mesh_path": self._step_artifact(job, "chart_stitch", "mesh") or "",
            "quad_mesh_path": self._step_artifact(job, "quad_remesh", "mesh") or "",
            "projected_quad_mesh_path": self._step_artifact(job, "feature_projection", "mesh") or "",
            "mesh_passport_path": self._step_artifact(job, "mesh_passport", "report") or "",
            "edit_image_path": self._resolve_metadata_local_uri(job, "edit_image_uri") or str(job.request.metadata.get("edit_image_path", "")),
        }
        values.update({key: str(value) for key, value in job.request.metadata.items() if isinstance(value, (str, int, float, bool))})
        if extra:
            values.update(extra)
        return values

    def _step_artifact(self, job: JobRecord, step_name: str, artifact_name: str) -> str | None:
        for step in job.steps:
            if step.name == step_name:
                return step.artifacts.get(artifact_name)
        return None

    def _resolve_metadata_local_uri(self, job: JobRecord, key: str) -> str | None:
        uri = job.request.metadata.get(key)
        if not isinstance(uri, str):
            return None
        path = self.artifacts.resolve_local_uri(uri)
        return str(path) if path is not None else None

    def _metadata_bool(self, job_id: str, key: str, *, default: bool = False) -> bool:
        job = self.store.get_job(job_id)
        value = job.request.metadata.get(key, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _production_gate_decision(
    metadata: dict[str, Any],
    mesh_metrics: dict[str, Any],
    quad_stats: dict[str, Any],
    blender_report: dict[str, Any] | None,
) -> dict[str, Any]:
    max_components = int(metadata.get("production_max_components", 1))
    max_boundary_loops = int(metadata.get("production_max_boundary_loops", 0))
    max_nonmanifold_edges = int(metadata.get("production_max_nonmanifold_edges", 0))
    max_nonmanifold_vertices = int(metadata.get("production_max_nonmanifold_vertices", 0))
    min_quad_ratio = float(metadata.get("production_min_quad_ratio", 0.85))
    require_blender = _bool_value(metadata.get("production_require_blender", False))

    metrics_ok = bool(mesh_metrics.get("ok"))
    watertight_ready = bool(
        metrics_ok
        and bool(mesh_metrics.get("watertight"))
        and int(mesh_metrics.get("connected_components") or 0) <= max_components
        and int(mesh_metrics.get("boundary_loop_count") or 0) <= max_boundary_loops
        and int(mesh_metrics.get("nonmanifold_edge_count") or 0) <= max_nonmanifold_edges
        and int(mesh_metrics.get("nonmanifold_vertex_count") or 0) <= max_nonmanifold_vertices
    )
    quad_ratio = float(quad_stats.get("quad_ratio") or 0.0)
    editable_quad_ready = bool(watertight_ready and quad_ratio >= min_quad_ratio)
    blender_skipped = blender_report is not None and bool(blender_report.get("skipped"))
    blender_ready = blender_report is None or bool(blender_report.get("ok")) or (blender_skipped and not require_blender)
    promoted = bool(watertight_ready and blender_ready)

    if editable_quad_ready and blender_ready:
        tier = "tier_3_editable_watertight_quad"
    elif watertight_ready and blender_ready:
        tier = "tier_1_watertight_triangle"
    elif metrics_ok:
        tier = "preview_or_repair_required"
    else:
        tier = "failed_validation"

    reasons: list[str] = []
    if not metrics_ok:
        reasons.append(str(mesh_metrics.get("error", "mesh_metrics_failed")))
    if not bool(mesh_metrics.get("watertight")):
        reasons.append("not_watertight")
    if int(mesh_metrics.get("connected_components") or 0) > max_components:
        reasons.append("too_many_components")
    if int(mesh_metrics.get("boundary_loop_count") or 0) > max_boundary_loops:
        reasons.append("open_boundaries")
    if int(mesh_metrics.get("nonmanifold_edge_count") or 0) > max_nonmanifold_edges:
        reasons.append("nonmanifold_edges")
    if int(mesh_metrics.get("nonmanifold_vertex_count") or 0) > max_nonmanifold_vertices:
        reasons.append("nonmanifold_vertices")
    if require_blender and (blender_report is None or not blender_report.get("ok")):
        reasons.append("required_blender_gates_failed_or_skipped")
    if watertight_ready and quad_ratio < min_quad_ratio:
        reasons.append("watertight_but_not_quad_editable")

    return {
        "promoted": promoted,
        "tier": tier,
        "watertight_ready": watertight_ready,
        "editable_quad_ready": editable_quad_ready,
        "quad_ratio": quad_ratio,
        "min_quad_ratio": min_quad_ratio,
        "blender_ready": blender_ready,
        "blender_required": require_blender,
        "reasons": reasons,
    }


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
