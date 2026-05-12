"""Artifact path helpers for production-style job outputs."""

from __future__ import annotations

import shutil
from pathlib import Path


class ArtifactStore:
    def __init__(self, root: str | Path = "artifacts"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def job_root(self, project_id: str, job_id: str) -> Path:
        return self.root / "projects" / project_id / "jobs" / job_id

    def path(self, project_id: str, job_id: str, *parts: str) -> Path:
        path = self.job_root(project_id, job_id).joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put_file(self, source: str | Path, project_id: str, job_id: str, *parts: str) -> Path:
        destination = self.path(project_id, job_id, *parts)
        shutil.copy2(source, destination)
        return destination

    def signed_url_placeholder(self, path: str | Path) -> str:
        return str(Path(path).resolve())

    def resolve_asset_path(self, uri: str | Path) -> Path:
        """Resolve a stored asset URI without allowing artifact-root escapes."""

        raw = str(uri)
        if raw.startswith("local://"):
            path = self.resolve_local_uri(raw)
            if path is None:
                raise ValueError(f"could not resolve local asset URI: {raw}")
            return path

        path = Path(raw).expanduser().resolve()
        root = self.root.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"asset path escapes artifact root: {raw}") from exc
        return path

    def resolve_local_uri(self, uri: str) -> Path | None:
        if not uri.startswith("local://"):
            return None
        relative = uri.removeprefix("local://").lstrip("/")
        path = (self.root / relative).resolve()
        root = self.root.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"local URI escapes artifact root: {uri}") from exc
        return path
