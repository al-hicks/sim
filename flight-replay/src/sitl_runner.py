from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class SitlRunner(ABC):
    @abstractmethod
    def run(
        self,
        mission_path: str | Path,
        params_path: str | Path,
        simulation_manifest: dict[str, Any] | None,
        output_path: str | Path,
    ) -> Path:
        """Run a replay backend and return output telemetry Parquet path."""


class SubprocessSitlRunner(SitlRunner):
    """Reference runner with fixture fallback for MVP scaffolding."""

    def __init__(self, sitl_binary: str | None = None) -> None:
        self.sitl_binary = sitl_binary

    def run(
        self,
        mission_path: str | Path,
        params_path: str | Path,
        simulation_manifest: dict[str, Any] | None,
        output_path: str | Path,
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = simulation_manifest or {}

        fixture_parquet = manifest.get("fixture_parquet")
        if fixture_parquet:
            fixture = Path(fixture_parquet)
            pd.read_parquet(fixture).to_parquet(out, index=False)
            return out

        if self.sitl_binary:
            self._invoke_binary(self.sitl_binary, mission_path, params_path, manifest)

        # MVP fallback for environments without SITL binaries.
        pd.DataFrame(
            {
                "t_us": [0, 100_000, 200_000],
                "t_mission_s": [0.0, 0.1, 0.2],
                "lat": [0.0, 0.0, 0.0],
                "lon": [0.0, 0.0, 0.0],
                "alt_m": [0.0, 0.1, 0.2],
                "vx": [0.0, 1.0, 1.0],
                "vy": [0.0, 0.0, 0.0],
                "vz": [0.0, 1.0, 1.0],
                "roll": [0.0, 0.0, 0.0],
                "pitch": [0.0, 0.0, 0.0],
                "yaw": [0.0, 0.0, 0.0],
                "batt_v": [16.0, 15.9, 15.8],
                "batt_a": [5.0, 6.0, 6.0],
                "mode": ["AUTO", "AUTO", "AUTO"],
                "armed": [True, True, False],
                "ekf_flags": [0, 0, 0],
                "mission_item_reached_seq": [1, None, 2],
                "phase": ["takeoff", "transition", "land"],
            }
        ).to_parquet(out, index=False)
        return out

    def _invoke_binary(
        self,
        sitl_binary: str,
        mission_path: str | Path,
        params_path: str | Path,
        manifest: dict[str, Any],
    ) -> None:
        cmd = [sitl_binary, "--mission", str(mission_path), "--params", str(params_path)]
        cmd.extend(["--manifest-json", json.dumps(manifest)])
        subprocess.run(cmd, check=True)
