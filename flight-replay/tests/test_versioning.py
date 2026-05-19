from __future__ import annotations

from pathlib import Path

from src.versioning import build_run_manifest, sha256_file


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "sample.txt"
    p.write_text("hello", encoding="utf-8")
    digest = sha256_file(p)
    assert digest is not None
    assert len(digest) == 64


def test_build_run_manifest_contains_expected_keys(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("thresholds: {}\n", encoding="utf-8")
    manifest = build_run_manifest(
        run_id="r1",
        started_at="2020-01-01T00:00:00+00:00",
        finished_at="2020-01-01T00:01:00+00:00",
        repo_root=tmp_path,
        config_path=config,
        simulation_manifest={"wind": {"speed_mps": 2.0}},
    )
    expected = {
        "run_id",
        "started_at",
        "finished_at",
        "git_commit",
        "ardupilot_version",
        "params_sha256",
        "mission_sha256",
        "input_log_sha256",
        "config_sha256",
        "simulation_manifest_sha256",
    }
    assert expected.issubset(manifest.keys())
