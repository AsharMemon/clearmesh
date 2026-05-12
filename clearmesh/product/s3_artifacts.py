"""S3-compatible artifact store for production deployments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore


class S3ArtifactStore(ArtifactStore):
    """Artifact store that mirrors local writes to S3/R2/MinIO.

    The worker still uses local paths while commands execute. Finished files can
    be uploaded and assets can store the returned `s3://...` URI. This keeps GPU
    adapters simple and makes the storage migration incremental.
    """

    def __init__(
        self,
        root: str | Path = "artifacts",
        *,
        bucket: str,
        prefix: str = "",
        client: Any | None = None,
    ) -> None:
        super().__init__(root)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is None:
            import boto3

            client = boto3.client("s3")
        self.client = client

    def object_key(self, project_id: str, job_id: str, *parts: str) -> str:
        key = "/".join(["projects", project_id, "jobs", job_id, *[part.strip("/") for part in parts if part]])
        return f"{self.prefix}/{key}" if self.prefix else key

    def upload_file(self, source: str | Path, project_id: str, job_id: str, *parts: str) -> str:
        key = self.object_key(project_id, job_id, *parts)
        self.client.upload_file(str(source), self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def signed_url_placeholder(self, path: str | Path) -> str:
        raw = str(path)
        if raw.startswith("s3://"):
            _, rest = raw.split("s3://", 1)
            bucket, key = rest.split("/", 1)
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=3600,
            )
        return super().signed_url_placeholder(path)
