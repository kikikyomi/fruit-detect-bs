from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import settings


logger = logging.getLogger(__name__)


@dataclass
class Record:
    id: int
    type: str
    created_at: str
    file_name: str
    input_path: str
    input_url: str
    output_path: str | None
    output_url: str | None
    result_json: str
    summary_json: str | None


_lock = Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_record(row: sqlite3.Row) -> Record:
    return Record(
        id=row["id"],
        type=row["type"],
        created_at=row["created_at"],
        file_name=row["file_name"],
        input_path=row["input_path"],
        input_url=row["input_url"],
        output_path=row["output_path"],
        output_url=row["output_url"],
        result_json=row["result_json"],
        summary_json=row["summary_json"],
    )


def _resolve_data_path(value: str | None) -> Path | None:
    if not value:
        return None

    path = Path(value)
    if not path.is_absolute():
        path = (settings.BACKEND_DIR / path).resolve()
    else:
        path = path.resolve()

    try:
        path.relative_to(settings.DATA_DIR.resolve())
    except Exception:
        return None
    return path


def _url_to_path(url: str | None) -> Path | None:
    if not url:
        return None
    if url.startswith("/static/outputs/"):
        return _resolve_data_path(str(settings.OUTPUT_DIR / url.replace("/static/outputs/", "", 1)))
    if url.startswith("/static/uploads/"):
        return _resolve_data_path(str(settings.UPLOAD_DIR / url.replace("/static/uploads/", "", 1)))
    return None


def _collect_summary_paths(summary_json: str | None) -> set[Path]:
    if not summary_json:
        return set()

    try:
        summary = json.loads(summary_json)
    except Exception:
        return set()

    paths: set[Path] = set()

    for keyframe_url in summary.get("keyframes", []):
        path = _url_to_path(keyframe_url if isinstance(keyframe_url, str) else None)
        if path is not None:
            paths.add(path)

    for item in summary.get("keyframe_details", []):
        if not isinstance(item, dict):
            continue
        image_url = item.get("image_url")
        path = _url_to_path(image_url if isinstance(image_url, str) else None)
        if path is not None:
            paths.add(path)

    return paths


def _collect_record_paths(record: Record) -> list[Path]:
    paths: set[Path] = set()

    input_path = _resolve_data_path(record.input_path)
    if input_path is not None:
        paths.add(input_path)

    output_path = _resolve_data_path(record.output_path)
    if output_path is not None:
        paths.add(output_path)

    paths.update(_collect_summary_paths(record.summary_json))
    return sorted(paths, key=str)


def _remove_file(path: Path | None) -> bool:
    if path is None or not path.exists() or not path.is_file():
        return False
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.warning("Failed to remove file during retention cleanup: %s (%s)", path, exc)
        return False


def _delete_record_row(conn: sqlite3.Connection, row: sqlite3.Row) -> tuple[int, int]:
    record = _row_to_record(row)
    removed_files = 0
    for path in _collect_record_paths(record):
        if _remove_file(path):
            removed_files += 1

    cursor = conn.execute("DELETE FROM records WHERE id = ?", (record.id,))
    return cursor.rowcount, removed_files


def init_db() -> None:
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    input_url TEXT NOT NULL,
                    output_path TEXT,
                    output_url TEXT,
                    result_json TEXT NOT NULL,
                    summary_json TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def create_record(
    record_type: str,
    file_name: str,
    input_path: Path,
    input_url: str,
    output_path: Path | None,
    output_url: str | None,
    result: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> int:
    created_at = datetime.now().isoformat(timespec="seconds")

    with _lock:
        conn = _connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO records (
                    type, created_at, file_name, input_path, input_url,
                    output_path, output_url, result_json, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_type,
                    created_at,
                    file_name,
                    str(input_path),
                    input_url,
                    str(output_path) if output_path else None,
                    output_url,
                    json.dumps(result, ensure_ascii=False),
                    json.dumps(summary, ensure_ascii=False) if summary else None,
                ),
            )
            conn.commit()
            record_id = int(cursor.lastrowid)
        finally:
            conn.close()

    if settings.RECORD_RETENTION_LIMIT > 0:
        prune_records(settings.RECORD_RETENTION_LIMIT)

    return record_id


def list_records(record_type: str, page: int, page_size: int) -> dict[str, Any]:
    offset = (page - 1) * page_size

    with _lock:
        conn = _connect()
        try:
            total = conn.execute("SELECT COUNT(1) FROM records WHERE type = ?", (record_type,)).fetchone()[0]
            rows = conn.execute(
                """
                SELECT * FROM records
                WHERE type = ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (record_type, page_size, offset),
            ).fetchall()
        finally:
            conn.close()

    items: list[dict[str, Any]] = []
    for row in rows:
        result = json.loads(row["result_json"]) if row["result_json"] else {}
        summary = json.loads(row["summary_json"]) if row["summary_json"] else {}

        boxes = result.get("boxes", []) if isinstance(result, dict) else []
        detection_count = len(boxes)
        if record_type == "video":
            detection_count = int(summary.get("detections", detection_count))

        items.append(
            {
                "id": row["id"],
                "type": row["type"],
                "created_at": row["created_at"],
                "file_name": row["file_name"],
                "input_url": row["input_url"],
                "output_url": row["output_url"],
                "stream_url": f"/api/detect/video/stream/{row['id']}" if record_type == "video" and row["output_url"] else None,
                "result": result,
                "summary": summary,
                "detection_count": detection_count,
            }
        )

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items,
    }


def get_record(record_id: int) -> Record | None:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
        finally:
            conn.close()

    if row is None:
        return None

    return _row_to_record(row)


def delete_record(record_id: int) -> bool:
    with _lock:
        conn = _connect()
        try:
            cursor = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


def delete_record_with_files(record_id: int) -> bool:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
            if row is None:
                return False

            deleted_records, _ = _delete_record_row(conn, row)
            conn.commit()
            return deleted_records > 0
        finally:
            conn.close()


def prune_records(limit: int) -> dict[str, int]:
    if limit <= 0:
        return {"deleted_records": 0, "deleted_files": 0}

    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM records
                ORDER BY id DESC
                LIMIT -1 OFFSET ?
                """,
                (limit,),
            ).fetchall()

            deleted_records = 0
            deleted_files = 0
            for row in rows:
                removed_records, removed_files = _delete_record_row(conn, row)
                deleted_records += removed_records
                deleted_files += removed_files

            conn.commit()
            return {"deleted_records": deleted_records, "deleted_files": deleted_files}
        finally:
            conn.close()


def cleanup_orphan_files() -> dict[str, int]:
    referenced_paths: set[Path] = set()

    with _lock:
        conn = _connect()
        try:
            rows = conn.execute("SELECT * FROM records").fetchall()
        finally:
            conn.close()

    for row in rows:
        referenced_paths.update(_collect_record_paths(_row_to_record(row)))

    removed_files = 0
    for base_dir in (settings.UPLOAD_DIR, settings.OUTPUT_DIR):
        for path in base_dir.iterdir():
            if not path.is_file():
                continue
            resolved_path = path.resolve()
            if resolved_path in referenced_paths:
                continue
            if _remove_file(resolved_path):
                removed_files += 1

    return {"deleted_files": removed_files}


def apply_storage_retention(clean_orphans: bool = False) -> dict[str, int]:
    deleted_records = 0
    deleted_files = 0

    if settings.RECORD_RETENTION_LIMIT > 0:
        prune_result = prune_records(settings.RECORD_RETENTION_LIMIT)
        deleted_records += prune_result["deleted_records"]
        deleted_files += prune_result["deleted_files"]

    if clean_orphans:
        cleanup_result = cleanup_orphan_files()
        deleted_files += cleanup_result["deleted_files"]

    return {"deleted_records": deleted_records, "deleted_files": deleted_files}
