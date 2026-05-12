"""Postgres-backed job store scaffold.

This mirrors `JsonJobStore` behavior with a compact JSONB schema. Migrations are
kept explicit in `schema_sql()` so prod-lite can bootstrap without adding a full
migration framework yet.
"""

from __future__ import annotations

import json
from typing import Any

from .models import JobRecord, JobStatus
from .store import JsonJobStore


def schema_sql() -> str:
    return """
CREATE TABLE IF NOT EXISTS clearmesh_jobs (
  id TEXT PRIMARY KEY,
  team_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS clearmesh_jobs_team_created_idx
  ON clearmesh_jobs (team_id, created_at DESC);
CREATE INDEX IF NOT EXISTS clearmesh_jobs_status_created_idx
  ON clearmesh_jobs (status, created_at ASC);
"""


class PostgresJobStore(JsonJobStore):
    """Drop-in store for API/worker processes once Postgres is configured."""

    def __init__(self, dsn: str):
        import psycopg

        self.dsn = dsn
        self.psycopg = psycopg
        with self.psycopg.connect(self.dsn) as conn:
            conn.execute(schema_sql())

    def create_job(self, job: JobRecord) -> JobRecord:
        payload = job.to_dict()
        with self.psycopg.connect(self.dsn) as conn:
            conn.execute(
                """
                INSERT INTO clearmesh_jobs (id, team_id, user_id, status, created_at, updated_at, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (job.id, job.team_id, job.user_id, job.status.value, job.created_at, job.updated_at, json.dumps(payload)),
            )
        return job

    def get_job(self, job_id: str) -> JobRecord:
        with self.psycopg.connect(self.dsn) as conn:
            row = conn.execute("SELECT payload FROM clearmesh_jobs WHERE id = %s", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._decode_job(dict(row[0]))

    def list_jobs(self, team_id: str | None = None) -> list[JobRecord]:
        if team_id is None:
            sql = "SELECT payload FROM clearmesh_jobs ORDER BY created_at DESC"
            params: tuple[Any, ...] = ()
        else:
            sql = "SELECT payload FROM clearmesh_jobs WHERE team_id = %s ORDER BY created_at DESC"
            params = (team_id,)
        with self.psycopg.connect(self.dsn) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode_job(dict(row[0])) for row in rows]

    def claim_next_queued(self) -> JobRecord | None:
        with self.psycopg.connect(self.dsn) as conn:
            with conn.transaction():
                row = conn.execute(
                    """
                    SELECT payload FROM clearmesh_jobs
                    WHERE status = %s
                    ORDER BY created_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                    """,
                    (JobStatus.QUEUED.value,),
                ).fetchone()
                if row is None:
                    return None
                job = self._decode_job(dict(row[0]))
                job.status = JobStatus.RUNNING
                self._upsert_job(conn, job)
                return job

    def update_job(self, job: JobRecord) -> JobRecord:
        from .models import utc_now

        job.updated_at = utc_now()
        with self.psycopg.connect(self.dsn) as conn:
            self._upsert_job(conn, job)
        return job

    def _upsert_job(self, conn: Any, job: JobRecord) -> None:
        conn.execute(
            """
            UPDATE clearmesh_jobs
            SET team_id = %s, user_id = %s, status = %s, updated_at = %s, payload = %s
            WHERE id = %s
            """,
            (job.team_id, job.user_id, job.status.value, job.updated_at, json.dumps(job.to_dict()), job.id),
        )
