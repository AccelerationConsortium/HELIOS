from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import uuid

from app.core.config import get_settings


def persist_json_artifact(run_id: str, step_id: str, payload: dict) -> tuple[str, str]:
    settings = get_settings()
    root = Path(settings.object_store_dir)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_id = str(uuid.uuid4())
    target = run_dir / f"{artifact_id}.json"

    raw = json.dumps(payload, sort_keys=True, indent=2)
    target.write_text(raw, encoding="utf-8")

    checksum = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return str(target), checksum


def persist_file_artifact(
    run_id: str,
    step_id: str,
    source_path: str | Path,
    *,
    suffix: str | None = None,
) -> tuple[str, str]:
    """Copy a file (CSV, PNG, etc.) into the object store and return (uri, sha256).

    Args:
        run_id: Owning run identifier.
        step_id: Owning step identifier (used for provenance, not in path).
        source_path: Absolute path to the file to ingest.
        suffix: Optional override for file extension (e.g. ``.csv``).
            Defaults to the source file's suffix.

    Returns:
        ``(artifact_uri, sha256_hex)``
    """
    settings = get_settings()
    root = Path(settings.object_store_dir)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    src = Path(source_path)
    ext = suffix or src.suffix or ""
    artifact_id = str(uuid.uuid4())
    target = run_dir / f"{artifact_id}{ext}"

    shutil.copy2(src, target)

    sha = hashlib.sha256()
    with open(target, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            sha.update(chunk)

    return str(target), sha.hexdigest()
