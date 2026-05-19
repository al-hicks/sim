from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class AlignmentError(ValueError):
    pass


@dataclass
class AnchorInfo:
    anchor_type: str
    anchor_t_us: int
    offset_from_arming_s: float | None


@dataclass
class AlignmentResult:
    real: pd.DataFrame
    sitl: pd.DataFrame
    metadata: dict[str, Any]


def discover_anchor(frame: pd.DataFrame) -> AnchorInfo | None:
    if frame.empty:
        return None
    work = frame.sort_values("t_us")

    reached = pd.to_numeric(
        work.get("mission_item_reached_seq", pd.Series(np.nan, index=work.index)),
        errors="coerce",
    )
    reached_seq1 = work[reached.eq(1)]
    if not reached_seq1.empty:
        anchor_t = int(reached_seq1["t_us"].iloc[0])
        return AnchorInfo(
            anchor_type="mission_item_reached",
            anchor_t_us=anchor_t,
            offset_from_arming_s=_offset_from_arming(work, anchor_t),
        )

    mode = work.get("mode", pd.Series("", index=work.index)).astype("string").str.upper()
    armed = work.get("armed", pd.Series(False, index=work.index)).fillna(False).astype(bool)
    entered_auto = mode.eq("AUTO") & armed & ~mode.shift(1, fill_value="").eq("AUTO")
    auto_rows = work[entered_auto]
    if not auto_rows.empty:
        anchor_t = int(auto_rows["t_us"].iloc[0])
        return AnchorInfo(
            anchor_type="auto_mode_entry",
            anchor_t_us=anchor_t,
            offset_from_arming_s=_offset_from_arming(work, anchor_t),
        )
    return None


def _offset_from_arming(frame: pd.DataFrame, anchor_t_us: int) -> float | None:
    armed = frame.get("armed", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    if not armed.any():
        return None
    first_armed = int(pd.to_numeric(frame.loc[armed, "t_us"], errors="coerce").iloc[0])
    return (anchor_t_us - first_armed) / 1e6


def align_streams(real: pd.DataFrame, sitl: pd.DataFrame, resample_hz: int = 10) -> AlignmentResult:
    real_anchor = discover_anchor(real)
    sitl_anchor = discover_anchor(sitl)
    if real_anchor is None or sitl_anchor is None:
        raise AlignmentError("alignment_anchor_not_found")

    real_norm = _anchor_and_resample(real, real_anchor.anchor_t_us, resample_hz)
    sitl_norm = _anchor_and_resample(sitl, sitl_anchor.anchor_t_us, resample_hz)
    overlap_real, overlap_sitl = _overlap_window(real_norm, sitl_norm, resample_hz)
    if overlap_real.empty or overlap_sitl.empty:
        raise AlignmentError("no_overlapping_samples_after_alignment")

    metadata = {
        "resample_hz": resample_hz,
        "real_anchor_type": real_anchor.anchor_type,
        "sitl_anchor_type": sitl_anchor.anchor_type,
        "real_anchor_offset_from_arming_s": real_anchor.offset_from_arming_s,
        "sitl_anchor_offset_from_arming_s": sitl_anchor.offset_from_arming_s,
    }
    return AlignmentResult(real=overlap_real, sitl=overlap_sitl, metadata=metadata)


def _anchor_and_resample(frame: pd.DataFrame, anchor_t_us: int, hz: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.sort_values("t_us").drop_duplicates("t_us").copy()
    work["t_mission_s"] = (pd.to_numeric(work["t_us"], errors="coerce") - anchor_t_us) / 1e6
    work = work.dropna(subset=["t_mission_s"])
    if work.empty:
        return work
    work = work.set_index("t_mission_s")

    start = float(work.index.min())
    end = float(work.index.max())
    if end <= start:
        raise AlignmentError("invalid_time_range_for_resampling")
    step = 1.0 / hz
    grid = np.arange(start, end + 1e-9, step)

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in work.columns:
        if col == "t_us":
            numeric_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(work[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    out = pd.DataFrame(index=grid)
    for col in numeric_cols:
        series = pd.to_numeric(work[col], errors="coerce").dropna()
        if series.empty:
            out[col] = np.nan
            continue
        out[col] = np.interp(grid, series.index.to_numpy(), series.to_numpy())

    for col in categorical_cols:
        sampled = work[col].reindex(work.index.union(grid)).sort_index().ffill().bfill()
        out[col] = sampled.reindex(grid).to_numpy()

    if "armed" in out.columns:
        out["armed"] = out["armed"].astype(bool)
    if "mission_item_reached_seq" in out.columns:
        out["mission_item_reached_seq"] = out["mission_item_reached_seq"].round()
    out["t_mission_s"] = out.index
    return out.reset_index(drop=True)


def _overlap_window(real: pd.DataFrame, sitl: pd.DataFrame, hz: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    t_r = pd.to_numeric(real["t_mission_s"], errors="coerce")
    t_s = pd.to_numeric(sitl["t_mission_s"], errors="coerce")
    start = max(float(t_r.min()), float(t_s.min()))
    end = min(float(t_r.max()), float(t_s.max()))
    if end <= start:
        return real.iloc[0:0], sitl.iloc[0:0]
    step = 1.0 / hz
    grid = np.arange(start, end + 1e-9, step)
    return _reindex_to_grid(real, grid), _reindex_to_grid(sitl, grid)


def _reindex_to_grid(frame: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    work = frame.set_index("t_mission_s")
    work = work.reindex(work.index.union(grid)).sort_index().interpolate(method="linear")
    out = work.reindex(grid).ffill().bfill()
    out["t_mission_s"] = out.index
    return out.reset_index(drop=True)
