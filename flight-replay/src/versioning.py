from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(data: dict[str, Any] | None) -> str | None:
    if data is None:
        return None
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_git_commit(cwd: str | Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def build_run_manifest(
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    repo_root: str | Path,
    ardupilot_version: str | None = None,
    params_path: str | Path | None = None,
    mission_path: str | Path | None = None,
    input_log_path: str | Path | None = None,
    config_path: str | Path | None = None,
    simulation_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "git_commit": get_git_commit(repo_root),
        "ardupilot_version": ardupilot_version,
        "params_sha256": sha256_file(params_path),
        "mission_sha256": sha256_file(mission_path),
        "input_log_sha256": sha256_file(input_log_path),
        "config_sha256": sha256_file(config_path),
        "simulation_manifest_sha256": sha256_json(simulation_manifest),
    }
