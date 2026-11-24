"""Persistent handoff between Studio, the dialogue worker, and the AD worker."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


LEASE_SECONDS = 30
MAX_ATTEMPTS = 2


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "studio-jobs.sqlite3"
        deadline = time.monotonic() + 10
        while True:
            try:
                self.initialize()
                break
            except sqlite3.OperationalError as error:
                # Simultaneous startup can race when switching a new database to WAL.
                locked = "locked" in str(error) or "busy" in str(error)
                if not locked or time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)

    def initialize(self) -> None:
        with self.connect() as db:
            if db.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                db.execute("PRAGMA journal_mode = WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS media (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    media_id TEXT NOT NULL REFERENCES media(id),
                    title TEXT NOT NULL,
                    host TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    requested TEXT NOT NULL,
                    allow_fallback INTEGER NOT NULL,
                    whisper_device TEXT,
                    ad_device TEXT,
                    whisper_host TEXT,
                    ad_host TEXT,
                    stage TEXT NOT NULL DEFAULT 'whisper',
                    state TEXT NOT NULL DEFAULT 'queued',
                    created_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_token TEXT,
                    lease_until REAL,
                    attempt_dir TEXT,
                    result_dir TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_pending ON jobs(stage, state, created_at);
                CREATE TABLE IF NOT EXISTS dialogue_results (
                    job_id TEXT PRIMARY KEY REFERENCES jobs(id),
                    work_dir TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workers (
                    role TEXT PRIMARY KEY,
                    host TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    cuda INTEGER NOT NULL,
                    preparer INTEGER NOT NULL,
                    seen_at REAL NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db
        finally:
            db.close()

    def register_media(self, media_id: str, source: Path) -> None:
        try:
            relative = source.resolve().relative_to(self.root)
        except ValueError:
            raise ValueError("Media must be on the shared job volume.") from None
        with self.connect() as db:
            db.execute("INSERT INTO media VALUES (?, ?)", (media_id, str(relative)))

    def source_path(self, media_id: str) -> Path | None:
        with self.connect() as db:
            row = db.execute("SELECT source FROM media WHERE id = ?", (media_id,)).fetchone()
        return self.root / row["source"] if row else None

    def enqueue(self, media_id: str, title: str, host: str, profile: str,
                requested: str, allow_fallback: bool) -> dict:
        job_id = uuid4().hex
        with self.connect() as db:
            db.execute(
                """INSERT INTO jobs (id, media_id, title, host, profile, requested, allow_fallback, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, media_id, title, host, profile, requested, allow_fallback, time.time()),
            )
            return dict(db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def get(self, job_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def recent(self, limit: int = 50) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,),
            )]

    def dialogue(self, job_id: str) -> tuple[Path, dict] | None:
        with self.connect() as db:
            row = db.execute("SELECT work_dir, payload FROM dialogue_results WHERE job_id = ?", (job_id,)).fetchone()
        return (self.root / row["work_dir"], json.loads(row["payload"])) if row else None

    def report_worker(self, role: str, host: str, profile: str, cuda: bool, preparer: bool) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO workers VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(role) DO UPDATE SET host=excluded.host, profile=excluded.profile,
                       cuda=excluded.cuda, preparer=excluded.preparer, seen_at=excluded.seen_at""",
                (role, host, profile, cuda, preparer, time.time()),
            )

    def active_workers(self) -> dict:
        with self.connect() as db:
            return {row["role"]: dict(row) for row in db.execute(
                "SELECT * FROM workers WHERE seen_at > ?", (time.time() - LEASE_SECONDS,),
            )}

    def claim(
        self, role: str, cuda: bool, host: str, profiles: dict[str, str] | None = None,
    ) -> dict | None:
        now = time.time()
        with self.connect() as db:
            # A write lock keeps selection and claim atomic across worker processes.
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """UPDATE jobs SET state='failed', error='Worker stopped before completion.',
                          lease_token=NULL, lease_until=NULL
                   WHERE stage=? AND state='running' AND lease_until<=? AND attempts>=?""",
                (role, now, MAX_ATTEMPTS),
            )
            row = db.execute(
                """SELECT * FROM jobs WHERE stage=?
                   AND (state='queued' OR (state='running' AND lease_until<=?)) ORDER BY created_at LIMIT 1""",
                (role, now),
            ).fetchone()
            if row is None:
                return None
            if row["requested"] == "cuda" and not cuda and not row["allow_fallback"]:
                db.execute(
                    """UPDATE jobs SET state='failed', error=?, lease_token=NULL, lease_until=NULL WHERE id=?""",
                    (f"CUDA is unavailable on the {role} worker.", row["id"]),
                )
                return None
            device = "cuda" if cuda and row["requested"] != "cpu" else "cpu"
            profile = profiles[device] if role == "ad" and profiles else row["profile"]
            token = uuid4().hex
            directory = str(Path(row["id"]) / f"{role}-{token}")
            whisper_device = device if role == "whisper" else row["whisper_device"]
            ad_device = device if role == "ad" else row["ad_device"]
            whisper_host = host if role == "whisper" else row["whisper_host"]
            ad_host = host if role == "ad" else row["ad_host"]
            db.execute(
                """UPDATE jobs SET state='running', attempts=attempts+1, lease_token=?, lease_until=?,
                          attempt_dir=?, whisper_device=?, ad_device=?, whisper_host=?, ad_host=?,
                          profile=?, error=NULL WHERE id=?""",
                (token, now + LEASE_SECONDS, directory, whisper_device, ad_device, whisper_host, ad_host,
                 profile, row["id"]),
            )
            return {**dict(row), "attempt_dir": directory, "lease_token": token, "selected_device": device,
                    "profile": profile}

    def renew(self, job: dict) -> bool:
        with self.connect() as db:
            return db.execute(
                "UPDATE jobs SET lease_until=? WHERE id=? AND lease_token=?",
                (time.time() + LEASE_SECONDS, job["id"], job["lease_token"]),
            ).rowcount == 1

    def finish_dialogue(self, job: dict, dialogue: dict) -> bool:
        with self.connect() as db:
            updated = db.execute(
                """UPDATE jobs SET state='queued', stage='ad', attempts=0, lease_token=NULL, lease_until=NULL
                   WHERE id=? AND lease_token=? AND stage='whisper'""",
                (job["id"], job["lease_token"]),
            ).rowcount
            if updated:
                db.execute("INSERT INTO dialogue_results VALUES (?, ?, ?)", (
                    job["id"], str(Path(job["attempt_dir"]) / "work"), json.dumps(dialogue, ensure_ascii=False),
                ))
            return updated == 1

    def finish_ad(self, job: dict) -> bool:
        with self.connect() as db:
            return db.execute(
                """UPDATE jobs SET state='finished', stage='complete', result_dir=attempt_dir,
                          lease_token=NULL, lease_until=NULL WHERE id=? AND lease_token=? AND stage='ad'""",
                (job["id"], job["lease_token"]),
            ).rowcount == 1

    def fail(self, job: dict, error: str) -> None:
        with self.connect() as db:
            db.execute(
                """UPDATE jobs SET state='failed', error=?, lease_token=NULL, lease_until=NULL
                   WHERE id=? AND lease_token=?""",
                (error, job["id"], job["lease_token"]),
            )
