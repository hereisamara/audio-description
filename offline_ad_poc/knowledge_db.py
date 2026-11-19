"""Local timestamp-indexed evidence store for audio-description decisions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence


class EvidenceStore:
    """Persist and retrieve time-aligned evidence without an external service."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> EvidenceStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS video (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                path TEXT NOT NULL,
                duration REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transcript_segments (
                id INTEGER PRIMARY KEY,
                start REAL NOT NULL,
                end REAL NOT NULL,
                text TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS shots (
                shot_id INTEGER PRIMARY KEY,
                start REAL NOT NULL,
                end REAL NOT NULL,
                boundary_score REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scenes (
                scene_id INTEGER PRIMARY KEY,
                start REAL NOT NULL,
                end REAL NOT NULL,
                shot_ids_json TEXT NOT NULL,
                boundary_confidence REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS boundaries (
                id INTEGER PRIMARY KEY,
                timestamp REAL NOT NULL,
                kind TEXT NOT NULL,
                score REAL,
                details_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evidence_frames (
                id INTEGER PRIMARY KEY,
                gap_index INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                requested_timestamp REAL NOT NULL,
                path TEXT NOT NULL UNIQUE,
                shot_id INTEGER,
                scene_id INTEGER,
                salience REAL NOT NULL,
                role TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS decisions (
                decision_index INTEGER PRIMARY KEY,
                gap_start REAL NOT NULL,
                gap_end REAL NOT NULL,
                decision TEXT NOT NULL,
                description TEXT NOT NULL,
                reason TEXT NOT NULL,
                importance REAL NOT NULL,
                event_start REAL,
                event_end REAL,
                event_timing_source TEXT NOT NULL,
                evidence_timestamps_json TEXT NOT NULL,
                cue_start REAL,
                cue_end REAL,
                placement_method TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS transcript_time_idx
                ON transcript_segments(start, end);
            CREATE INDEX IF NOT EXISTS boundary_time_idx
                ON boundaries(timestamp);
            CREATE INDEX IF NOT EXISTS frame_time_idx
                ON evidence_frames(timestamp);
            """
        )
        self.connection.commit()

    def reset(self) -> None:
        """Clear a prior run when the same work directory is reused."""
        with self.connection:
            for table in (
                "decisions",
                "evidence_frames",
                "boundaries",
                "scenes",
                "shots",
                "transcript_segments",
                "video",
            ):
                self.connection.execute(f"DELETE FROM {table}")

    def index_timeline(
        self,
        video_path: Path,
        duration: float,
        segments: Sequence[Any],
        shots: Sequence[Any],
        scenes: Sequence[Any],
        shot_validation: dict[str, Any],
    ) -> None:
        """Write all non-VLM context that can support later gap queries."""
        self.reset()
        with self.connection:
            self.connection.execute(
                "INSERT INTO video(id, path, duration) VALUES (1, ?, ?)",
                (str(video_path), float(duration)),
            )
            self.connection.executemany(
                "INSERT INTO transcript_segments(id, start, end, text) VALUES (?, ?, ?, ?)",
                [
                    (index, float(item.start), float(item.end), str(item.text))
                    for index, item in enumerate(segments, start=1)
                ],
            )
            self.connection.executemany(
                "INSERT INTO shots(shot_id, start, end, boundary_score) VALUES (?, ?, ?, ?)",
                [
                    (
                        int(item.shot_id),
                        float(item.start),
                        float(item.end),
                        float(item.boundary_score),
                    )
                    for item in shots
                ],
            )
            self.connection.executemany(
                """
                INSERT INTO scenes(
                    scene_id, start, end, shot_ids_json, boundary_confidence
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(item.scene_id),
                        float(item.start),
                        float(item.end),
                        json.dumps(list(item.shot_ids)),
                        float(item.boundary_confidence),
                    )
                    for item in scenes
                ],
            )
            boundary_rows: list[tuple[float, str, float | None, str]] = []
            for item in shot_validation.get("evidence", []):
                boundary_rows.append(
                    (
                        float(item["timestamp"]),
                        "confirmed_cut",
                        float(item.get("pixel_difference", 0.0)),
                        json.dumps(item, sort_keys=True),
                    )
                )
            for item in shot_validation.get("soft_candidates_requiring_review", []):
                boundary_rows.append(
                    (
                        float(item["timestamp"]),
                        "soft_candidate",
                        float(item.get("frame_difference_score", 0.0)),
                        json.dumps(item, sort_keys=True),
                    )
                )
            self.connection.executemany(
                "INSERT INTO boundaries(timestamp, kind, score, details_json) VALUES (?, ?, ?, ?)",
                boundary_rows,
            )

    def record_frames(self, gap_index: int, frames: Iterable[Any]) -> None:
        with self.connection:
            self.connection.executemany(
                """
                INSERT OR REPLACE INTO evidence_frames(
                    gap_index, timestamp, requested_timestamp, path, shot_id,
                    scene_id, salience, role
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(gap_index),
                        float(frame.timestamp),
                        float(frame.requested_timestamp),
                        str(frame.path),
                        frame.shot_id,
                        frame.scene_id,
                        float(frame.salience),
                        str(frame.role),
                    )
                    for frame in frames
                ],
            )

    def query_gap_context(
        self,
        gap_start: float,
        gap_end: float,
        lookback_seconds: float = 8.0,
    ) -> dict[str, Any]:
        window_start = max(0.0, gap_start - lookback_seconds)
        transcript = [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT start, end, text FROM transcript_segments
                WHERE end > ? AND start < ? ORDER BY start
                """,
                (window_start, gap_end),
            )
        ]
        boundaries = [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT timestamp, kind, score FROM boundaries
                WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp
                """,
                (window_start, gap_end),
            )
        ]
        frames = [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT timestamp, path, shot_id, scene_id, salience, role
                FROM evidence_frames
                WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp
                """,
                (window_start, gap_end),
            )
        ]
        previous_ad = [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT event_start, event_end, description FROM decisions
                WHERE decision = 'AD' AND gap_end <= ? ORDER BY gap_end DESC LIMIT 3
                """,
                (gap_start,),
            )
        ]
        return {
            "window_start": window_start,
            "window_end": gap_end,
            "transcript": transcript,
            "boundaries": boundaries,
            "frames": frames,
            "previous_ad": list(reversed(previous_ad)),
        }

    def record_decision(self, decision_index: int, decision: Any) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO decisions(
                    decision_index, gap_start, gap_end, decision, description,
                    reason, importance, event_start, event_end,
                    event_timing_source, evidence_timestamps_json, cue_start,
                    cue_end, placement_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(decision_index),
                    float(decision.gap_start),
                    float(decision.gap_end),
                    str(decision.decision),
                    str(decision.description),
                    str(decision.reason),
                    float(decision.importance),
                    decision.event_start,
                    decision.event_end,
                    str(decision.event_timing_source),
                    json.dumps(list(decision.evidence_timestamps)),
                    decision.cue_start or None,
                    decision.cue_end or None,
                    str(decision.placement_method),
                ),
            )

