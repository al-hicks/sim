from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def write_metrics_json(metrics: dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return out


def generate_html_report(
    real: pd.DataFrame,
    sitl: pd.DataFrame,
    metrics: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
    out_path: str | Path,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "XY trajectory (local ENU m)",
            "Altitude over time",
            "Horizontal error over time",
            "Velocity comparison",
            "Motor outputs",
        ),
        vertical_spacing=0.05,
    )

    x_real, y_real = _latlon_to_enu(real, real)
    x_sitl, y_sitl = _latlon_to_enu(sitl, real)
    t = pd.to_numeric(real["t_mission_s"], errors="coerce")
    horizontal_error = np.sqrt((x_real - x_sitl) ** 2 + (y_real - y_sitl) ** 2)
    speed_real = _speed(real)
    speed_sitl = _speed(sitl)

    fig.add_trace(go.Scatter(x=x_real, y=y_real, mode="lines", name="real XY"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_sitl, y=y_sitl, mode="lines", name="sitl XY"), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=t, y=real["alt_m"], mode="lines", name="real alt (m)"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=sitl["alt_m"], mode="lines", name="sitl alt (m)"),
        row=2,
        col=1,
    )

    fig.add_trace(go.Scatter(x=t, y=horizontal_error, mode="lines", name="horizontal error"), row=3, col=1)
    _shade_phases(fig, t, real.get("phase", pd.Series("other", index=real.index)), row=3, col=1)

    fig.add_trace(go.Scatter(x=t, y=speed_real, mode="lines", name="real speed"), row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=speed_sitl, mode="lines", name="sitl speed"), row=4, col=1)

    rcout_cols = [c for c in sitl.columns if c.startswith("rcout_")]
    if rcout_cols:
        for col in rcout_cols:
            fig.add_trace(go.Scatter(x=t, y=sitl[col], mode="lines", name=col), row=5, col=1)
    else:
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode="markers", name="no RCOUT data"),
            row=5,
            col=1,
        )

    fig.update_layout(height=1500, title="Flight Replay Comparator Report")
    charts_html = fig.to_html(include_plotlyjs=True, full_html=False)
    metrics_table_html = _metrics_table(metrics)
    badges_html = _pass_fail_badges(metrics, thresholds)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Flight Replay Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 6px; margin: 2px; }}
    .pass {{ background: #d4edda; color: #155724; }}
    .fail {{ background: #f8d7da; color: #721c24; }}
    table {{ border-collapse: collapse; margin-top: 16px; min-width: 900px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>Flight Replay Comparator</h1>
  <h2>Threshold status</h2>
  {badges_html}
  <h2>Metrics</h2>
  {metrics_table_html}
  <h2>Plots</h2>
  {charts_html}
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out


def _latlon_to_enu(frame: pd.DataFrame, ref_frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    lat = pd.to_numeric(frame.get("lat"), errors="coerce")
    lon = pd.to_numeric(frame.get("lon"), errors="coerce")
    ref_lat = float(pd.to_numeric(ref_frame.get("lat"), errors="coerce").dropna().iloc[0])
    ref_lon = float(pd.to_numeric(ref_frame.get("lon"), errors="coerce").dropna().iloc[0])
    r = 6_378_137.0
    x = np.deg2rad(lon - ref_lon) * r * np.cos(np.deg2rad(ref_lat))
    y = np.deg2rad(lat - ref_lat) * r
    return pd.Series(x), pd.Series(y)


def _speed(frame: pd.DataFrame) -> pd.Series:
    vx = pd.to_numeric(frame.get("vx"), errors="coerce")
    vy = pd.to_numeric(frame.get("vy"), errors="coerce")
    vz = pd.to_numeric(frame.get("vz"), errors="coerce")
    return np.sqrt(vx**2 + vy**2 + vz**2)


def _shade_phases(fig: go.Figure, t: pd.Series, phase: pd.Series, *, row: int, col: int) -> None:
    phase = phase.astype("string").fillna("other")
    t = pd.to_numeric(t, errors="coerce")
    if phase.empty:
        return
    colors = {
        "takeoff": "rgba(0, 123, 255, 0.08)",
        "cruise": "rgba(40, 167, 69, 0.08)",
        "transition": "rgba(255, 193, 7, 0.12)",
        "loiter": "rgba(23, 162, 184, 0.10)",
        "land": "rgba(220, 53, 69, 0.08)",
        "other": "rgba(108, 117, 125, 0.06)",
    }
    start = float(t.iloc[0])
    current = str(phase.iloc[0])
    for i in range(1, len(phase)):
        if phase.iloc[i] != current:
            fig.add_vrect(
                x0=start,
                x1=float(t.iloc[i - 1]),
                fillcolor=colors.get(current, colors["other"]),
                line_width=0,
                row=row,
                col=col,
            )
            start = float(t.iloc[i])
            current = str(phase.iloc[i])
    fig.add_vrect(
        x0=start,
        x1=float(t.iloc[-1]),
        fillcolor=colors.get(current, colors["other"]),
        line_width=0,
        row=row,
        col=col,
    )


def _metrics_table(metrics: dict[str, Any]) -> str:
    units = metrics.get("units", {})
    rows: list[dict[str, Any]] = []
    phase_metrics = metrics.get("phases", {})
    overall = metrics.get("overall", {})
    metric_names = sorted({k for p in phase_metrics.values() for k in p.keys()} | set(overall.keys()))

    for name in metric_names:
        row: dict[str, Any] = {"metric": name, "unit": units.get(name, "-"), "overall": overall.get(name)}
        for phase, vals in phase_metrics.items():
            row[phase] = vals.get(name)
        rows.append(row)
    if not rows:
        return "<p>No metrics generated.</p>"
    table = pd.DataFrame(rows)
    return table.to_html(index=False, border=0)


def _pass_fail_badges(metrics: dict[str, Any], thresholds: dict[str, dict[str, float]]) -> str:
    parts: list[str] = []
    phase_metrics = metrics.get("phases", {})
    for phase, metric_thresholds in thresholds.items():
        observed = phase_metrics.get(phase, {})
        for metric_name, expected in metric_thresholds.items():
            actual = observed.get(metric_name)
            if actual is None:
                status = "fail"
                label = f"{phase}:{metric_name} unavailable"
            else:
                passed = float(actual) <= float(expected)
                status = "pass" if passed else "fail"
                label = f"{phase}:{metric_name} {actual:.3f} <= {expected:.3f}"
            parts.append(f'<span class="badge {status}">{label}</span>')
    return "".join(parts) if parts else "<p>No thresholds configured.</p>"
