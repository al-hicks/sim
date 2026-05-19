from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RejectionConfig

PHASES = ("takeoff", "cruise", "transition", "loiter", "land", "other")


def tag_flight_phases(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="string")

    work = frame.sort_values("t_us").copy()
    mode = work.get("mode", pd.Series("UNKNOWN", index=work.index)).fillna("UNKNOWN")
    mode_upper = mode.astype("string").str.upper()
    armed = work.get("armed", pd.Series(False, index=work.index)).fillna(False).astype(bool)
    vz = pd.to_numeric(work.get("vz", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0)

    reached = pd.to_numeric(
        work.get("mission_item_reached_seq", pd.Series(np.nan, index=work.index)),
        errors="coerce",
    )
    transition_mask = _transition_mask(work["t_us"], reached)
    first_waypoint_reached_idx = reached.first_valid_index()
    before_first_waypoint = (
        pd.Series(True, index=work.index)
        if first_waypoint_reached_idx is None
        else (work.index <= first_waypoint_reached_idx)
    )

    phase = pd.Series("other", index=work.index, dtype="string")
    phase[transition_mask] = "transition"

    takeoff_mask = armed & (vz > 0.4) & before_first_waypoint & (phase == "other")
    phase[takeoff_mask] = "takeoff"

    loiter_mask = mode_upper.eq("LOITER") & (phase == "other")
    phase[loiter_mask] = "loiter"

    land_mode = mode_upper.isin(["LAND", "RTL"])
    land_mask = land_mode & (vz < -0.3) & (phase == "other")
    phase[land_mask] = "land"

    cruise_mask = mode_upper.eq("AUTO") & (vz.abs() <= 1.0) & (phase == "other")
    phase[cruise_mask] = "cruise"

    return phase.reindex(frame.index)


def _transition_mask(t_us: pd.Series, reached_seq: pd.Series, seconds: float = 5.0) -> pd.Series:
    t = pd.to_numeric(t_us, errors="coerce")
    seq = pd.to_numeric(reached_seq, errors="coerce")
    mask = pd.Series(False, index=t_us.index)
    event_times = t[seq.notna()].to_numpy(dtype=float)
    if event_times.size == 0:
        return mask
    window_us = seconds * 1e6
    ts = t.to_numpy(dtype=float)
    for ev in event_times:
        mask |= pd.Series((ts >= ev) & (ts <= ev + window_us), index=t_us.index)
    return mask


@dataclass
class RejectionResult:
    rejected: bool
    reasons: list[str]


def detect_rejection(
    frame: pd.DataFrame,
    *,
    cfg: RejectionConfig,
    anchor_found: bool,
) -> RejectionResult:
    reasons: list[str] = []
    if frame.empty:
        reasons.append("empty_log")
        return RejectionResult(rejected=True, reasons=reasons)

    armed = frame.get("armed", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    t_us = pd.to_numeric(frame.get("t_us"), errors="coerce")

    if armed.any():
        first = t_us[armed].iloc[0]
        last = t_us[armed].iloc[-1]
        armed_seconds = float((last - first) / 1e6)
    else:
        armed_seconds = 0.0
    if armed_seconds < cfg.min_armed_seconds:
        reasons.append("armed_time_below_minimum")

    mode = frame.get("mode", pd.Series("", index=frame.index)).astype("string").str.upper()
    if not mode.eq("AUTO").any():
        reasons.append("mode_never_entered_auto")

    max_gap_pct = _critical_gap_pct(frame)
    if max_gap_pct > cfg.max_gap_pct:
        reasons.append("critical_stream_gaps_exceed_threshold")

    landed = mode.isin(["LAND", "RTL"]).any()
    disarmed_end = bool(~armed.iloc[-1]) if len(armed) else False
    if not landed and not disarmed_end:
        reasons.append("post_crash_truncation_possible")

    if not anchor_found:
        reasons.append("alignment_anchor_not_found")

    return RejectionResult(rejected=bool(reasons), reasons=reasons)


def _critical_gap_pct(frame: pd.DataFrame) -> float:
    critical_cols = {
        "GPS": ["lat", "lon", "alt_m"],
        "ATT": ["roll", "pitch", "yaw"],
        "CTUN": ["vx", "vy", "vz"],
    }
    gap_pcts: list[float] = []
    for cols in critical_cols.values():
        available_cols = [col for col in cols if col in frame.columns]
        if not available_cols:
            gap_pcts.append(100.0)
            continue
        missing = frame[available_cols].isna().all(axis=1).mean() * 100.0
        gap_pcts.append(float(missing))
    return max(gap_pcts) if gap_pcts else 100.0
