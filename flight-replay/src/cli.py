from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import typer

from .align import AlignmentError, align_streams
from .config import RuntimeConfig, load_config
from .log_extract import extract_log_to_parquet
from .metrics import compute_metrics
from .mission_extract import write_mission_json
from .phases import detect_rejection, tag_flight_phases
from .report import generate_html_report, write_metrics_json
from .sitl_runner import SubprocessSitlRunner
from .versioning import build_run_manifest, now_utc_iso

app = typer.Typer(no_args_is_help=True, help="ArduPilot SITL flight replay comparator")


@dataclass
class RunContext:
    run_id: str
    started_at: str
    run_dir: Path
    log_path: Path

    def log(self, level: str, event: str, **payload: Any) -> None:
        entry = {"ts": now_utc_iso(), "level": level, "event": event, **payload}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _start_run(reports_root: Path, command: str) -> RunContext:
    run_id = str(uuid.uuid4())
    run_dir = reports_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    ctx = RunContext(run_id=run_id, started_at=now_utc_iso(), run_dir=run_dir, log_path=log_path)
    ctx.log("info", "run_started", command=command, run_id=run_id)
    return ctx


def _finish_run(
    ctx: RunContext,
    *,
    repo_root: Path,
    ardupilot_version: str | None = None,
    params_path: Path | None = None,
    mission_path: Path | None = None,
    input_log_path: Path | None = None,
    config_path: Path | None = None,
    simulation_manifest: dict[str, Any] | None = None,
) -> Path:
    finished_at = now_utc_iso()
    manifest = build_run_manifest(
        run_id=ctx.run_id,
        started_at=ctx.started_at,
        finished_at=finished_at,
        repo_root=repo_root,
        ardupilot_version=ardupilot_version,
        params_path=params_path,
        mission_path=mission_path,
        input_log_path=input_log_path,
        config_path=config_path,
        simulation_manifest=simulation_manifest,
    )
    if simulation_manifest is not None:
        manifest["simulation_manifest"] = simulation_manifest
    manifest_path = ctx.run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ctx.log("info", "run_finished", manifest_path=str(manifest_path))
    return manifest_path


@app.command("extract-log")
def extract_log_cmd(
    input: Path = typer.Option(..., exists=True, help="Input ArduPilot .BIN file"),
    output: Path = typer.Option(..., help="Output extracted parquet path"),
    reports_dir: Path = typer.Option(
        Path("data/reports"),
        help="Root folder for per-run reports and structured logs",
    ),
) -> None:
    ctx = _start_run(reports_dir, "extract-log")
    ctx.log("info", "extract_log_started", input=str(input), output=str(output))
    out_path, meta = extract_log_to_parquet(input, output)
    ctx.log("info", "extract_log_completed", rows=meta.get("rows"), output=str(out_path))
    _finish_run(
        ctx,
        repo_root=Path.cwd(),
        ardupilot_version=meta.get("ardupilot_version"),
        input_log_path=input,
    )


@app.command("extract-mission")
def extract_mission_cmd(
    input: Path = typer.Option(..., exists=True, help="Input ArduPilot .BIN file"),
    output: Path = typer.Option(..., help="Output mission JSON path"),
    reports_dir: Path = typer.Option(Path("data/reports"), help="Root folder for run logs"),
) -> None:
    ctx = _start_run(reports_dir, "extract-mission")
    ctx.log("info", "extract_mission_started", input=str(input), output=str(output))
    mission_path = write_mission_json(input, output)
    ctx.log("info", "extract_mission_completed", output=str(mission_path))
    _finish_run(ctx, repo_root=Path.cwd(), input_log_path=input, mission_path=mission_path)


@app.command("run-sitl")
def run_sitl_cmd(
    mission: Path = typer.Option(..., exists=True, help="Mission JSON"),
    params: Path = typer.Option(..., exists=True, help="ArduPilot params .param file"),
    output: Path = typer.Option(..., help="Replay telemetry parquet output"),
    simulation_manifest_path: Path | None = typer.Option(
        None,
        help="Optional JSON manifest for wind/vehicle/battery/SIM_* overrides",
    ),
    sitl_binary: str | None = typer.Option(None, help="Optional SITL binary path"),
    reports_dir: Path = typer.Option(Path("data/reports"), help="Root folder for run logs"),
) -> None:
    simulation_manifest: dict[str, Any] | None = None
    if simulation_manifest_path:
        simulation_manifest = json.loads(simulation_manifest_path.read_text(encoding="utf-8"))

    ctx = _start_run(reports_dir, "run-sitl")
    ctx.log(
        "info",
        "run_sitl_started",
        mission=str(mission),
        params=str(params),
        output=str(output),
        simulation_manifest=simulation_manifest,
    )
    runner = SubprocessSitlRunner(sitl_binary=sitl_binary)
    replay_path = runner.run(mission, params, simulation_manifest, output)
    ctx.log("info", "run_sitl_completed", replay_path=str(replay_path))
    _finish_run(
        ctx,
        repo_root=Path.cwd(),
        params_path=params,
        mission_path=mission,
        simulation_manifest=simulation_manifest,
    )


@app.command("compare")
def compare_cmd(
    real: Path = typer.Option(..., exists=True, help="Real flight parquet"),
    sitl: Path = typer.Option(..., exists=True, help="SITL replay parquet"),
    mission: Path | None = typer.Option(None, help="Mission JSON"),
    config: Path = typer.Option(Path("config.yaml"), exists=True, help="Comparator config"),
    output_dir: Path = typer.Option(Path("data/reports"), help="Output reports root"),
) -> None:
    ctx = _start_run(output_dir, "compare")
    cfg = load_config(config)
    mission_payload = _load_mission(mission)
    ctx.log("info", "compare_started", real=str(real), sitl=str(sitl), config=str(config))

    real_df = pd.read_parquet(real)
    sitl_df = pd.read_parquet(sitl)
    if "phase" not in real_df.columns:
        real_df["phase"] = tag_flight_phases(real_df)
    if "phase" not in sitl_df.columns:
        sitl_df["phase"] = tag_flight_phases(sitl_df)

    try:
        aligned = align_streams(real_df, sitl_df, resample_hz=cfg.alignment.resample_hz)
        anchor_found = True
    except AlignmentError as exc:
        anchor_found = False
        aligned = None
        ctx.log("error", "alignment_failed", reason=str(exc))

    rejection = detect_rejection(real_df, cfg=cfg.rejection, anchor_found=anchor_found)
    if aligned is None:
        metrics = {
            "rejected": True,
            "rejection_reasons": rejection.reasons,
            "units": {},
            "overall": {},
            "phases": {},
            "mission": {},
            "informational": {},
            "alignment": {},
        }
    else:
        metrics = compute_metrics(
            aligned.real,
            aligned.sitl,
            mission=mission_payload,
            alignment_metadata=aligned.metadata,
            mot_pwm_max=cfg.sim.mot_pwm_max,
        )
        metrics["rejected"] = rejection.rejected
        metrics["rejection_reasons"] = rejection.reasons

    metrics_path = write_metrics_json(metrics, ctx.run_dir / "metrics.json")
    plots_path = ctx.run_dir / "plots.html"
    if aligned is not None:
        generate_html_report(aligned.real, aligned.sitl, metrics, cfg.thresholds, plots_path)
    else:
        plots_path.write_text("<html><body><h1>Alignment failed</h1></body></html>", encoding="utf-8")

    ctx.log(
        "info",
        "compare_completed",
        metrics_path=str(metrics_path),
        plots_path=str(plots_path),
        rejected=metrics.get("rejected"),
    )
    _finish_run(
        ctx,
        repo_root=Path.cwd(),
        mission_path=mission,
        config_path=config,
    )


@app.command("batch")
def batch_cmd(
    real_folder: Path = typer.Option(..., exists=True, help="Folder containing real parquet logs"),
    sitl_folder: Path = typer.Option(..., exists=True, help="Folder containing SITL parquet logs"),
    mission_folder: Path | None = typer.Option(None, help="Folder containing mission JSON files"),
    config: Path = typer.Option(Path("config.yaml"), exists=True, help="Comparator config"),
    output_dir: Path = typer.Option(Path("data/reports"), help="Batch output root"),
) -> None:
    ctx = _start_run(output_dir, "batch")
    cfg = load_config(config)
    rows: list[dict[str, Any]] = []

    for real_file in sorted(real_folder.glob("*.parquet")):
        stem = real_file.stem
        sitl_file = sitl_folder / f"{stem}.parquet"
        mission_file = mission_folder / f"{stem}.json" if mission_folder else None
        if not sitl_file.exists():
            ctx.log("warning", "missing_sitl_replay", flight=stem, expected=str(sitl_file))
            continue

        real_df = pd.read_parquet(real_file)
        sitl_df = pd.read_parquet(sitl_file)
        if "phase" not in real_df.columns:
            real_df["phase"] = tag_flight_phases(real_df)
        if "phase" not in sitl_df.columns:
            sitl_df["phase"] = tag_flight_phases(sitl_df)

        mission_payload = _load_mission(mission_file)
        try:
            aligned = align_streams(real_df, sitl_df, resample_hz=cfg.alignment.resample_hz)
            anchor_found = True
        except AlignmentError as exc:
            aligned = None
            anchor_found = False
            ctx.log("error", "batch_alignment_failed", flight=stem, reason=str(exc))

        rejection = detect_rejection(real_df, cfg=cfg.rejection, anchor_found=anchor_found)
        if aligned is None:
            metrics = {"rejected": True, "rejection_reasons": rejection.reasons, "overall": {}, "phases": {}}
        else:
            metrics = compute_metrics(
                aligned.real,
                aligned.sitl,
                mission=mission_payload,
                alignment_metadata=aligned.metadata,
                mot_pwm_max=cfg.sim.mot_pwm_max,
            )
            metrics["rejected"] = rejection.rejected
            metrics["rejection_reasons"] = rejection.reasons

        flight_metrics_path = ctx.run_dir / f"{stem}.metrics.json"
        write_metrics_json(metrics, flight_metrics_path)
        rows.append(
            {
                "flight": stem,
                "rejected": bool(metrics.get("rejected")),
                "rms_horizontal_error_m": metrics.get("overall", {}).get("rms_horizontal_error_m"),
                "rms_velocity_error_mps": metrics.get("overall", {}).get("rms_velocity_error_mps"),
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        con = duckdb.connect()
        con.register("flight_metrics", df)
        summary_df = con.execute(
            """
            select
              count(*) as flights_total,
              sum(case when rejected then 1 else 0 end) as flights_rejected,
              avg(rms_horizontal_error_m) as avg_rms_horizontal_error_m,
              avg(rms_velocity_error_mps) as avg_rms_velocity_error_mps
            from flight_metrics
            where rejected = false
            """
        ).fetchdf()
        per_flight_csv = ctx.run_dir / "summary.csv"
        df.to_csv(per_flight_csv, index=False)
        summary_html = ctx.run_dir / "summary.html"
        summary_html.write_text(
            "<h1>Batch Summary</h1>\n"
            + summary_df.to_html(index=False)
            + "\n<h2>Per-flight</h2>\n"
            + df.to_html(index=False),
            encoding="utf-8",
        )
        ctx.log("info", "batch_completed", flights=len(rows), summary_csv=str(per_flight_csv))
    else:
        ctx.log("warning", "batch_no_rows")

    _finish_run(ctx, repo_root=Path.cwd(), config_path=config)


def _load_mission(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
