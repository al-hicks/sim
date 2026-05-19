from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from src.cli import app
from tests.helpers import synthetic_real_sitl_frames


def test_golden_compare_regression(tmp_path: Path):
    real, sitl = synthetic_real_sitl_frames(duration_s=60, hz=10)
    real_path = tmp_path / "real.parquet"
    sitl_path = tmp_path / "sitl.parquet"
    real.to_parquet(real_path, index=False)
    sitl.to_parquet(sitl_path, index=False)

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    mission_path = project_root / "tests" / "fixtures" / "mission_fixture.json"
    reports_root = tmp_path / "reports"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "compare",
            "--real",
            str(real_path),
            "--sitl",
            str(sitl_path),
            "--mission",
            str(mission_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(reports_root),
        ],
    )
    assert result.exit_code == 0, result.stdout

    run_dirs = [p for p in reports_root.iterdir() if p.is_dir()]
    assert run_dirs, "no run directory was created"
    metrics = json.loads((run_dirs[0] / "metrics.json").read_text(encoding="utf-8"))
    golden = json.loads((project_root / "tests" / "fixtures" / "golden_metrics.json").read_text(encoding="utf-8"))

    assert (project_root / "tests" / "fixtures" / "anonymized_60s.BIN").exists()
    assert (run_dirs[0] / "run.log").exists()
    _assert_drift_within_pct(metrics, golden, limit_pct=2.0)


def _assert_drift_within_pct(actual: dict, golden: dict, limit_pct: float) -> None:
    for key in _flatten_numeric(golden):
        g = _get_nested(golden, key)
        a = _get_nested(actual, key)
        if g in (None, 0):
            continue
        assert a is not None, f"missing metric {key}"
        drift_pct = abs(float(a) - float(g)) / abs(float(g)) * 100.0
        assert drift_pct <= limit_pct, f"{key} drift {drift_pct:.3f}% > {limit_pct}%"


def _flatten_numeric(payload: dict, prefix: str = "") -> list[str]:
    keys: list[str] = []
    for k, v in payload.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys.extend(_flatten_numeric(v, path))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            keys.append(path)
    return keys


def _get_nested(payload: dict, path: str):
    value = payload
    for part in path.split("."):
        value = value.get(part)
        if value is None:
            return None
    return value
