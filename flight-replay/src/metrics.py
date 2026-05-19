from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from .phases import PHASES


@dataclass
class ErrorSeries:
    horizontal_error_m: pd.Series
    altitude_error_m: pd.Series
    velocity_error_mps: pd.Series
    attitude_error_deg: pd.Series
    real_speed_mps: pd.Series
    sitl_speed_mps: pd.Series
    x_real_m: pd.Series
    y_real_m: pd.Series
    x_sitl_m: pd.Series
    y_sitl_m: pd.Series


def compute_metrics(
    real: pd.DataFrame,
    sitl: pd.DataFrame,
    *,
    mission: dict[str, Any] | None = None,
    alignment_metadata: dict[str, Any] | None = None,
    mot_pwm_max: int = 2000,
) -> dict[str, Any]:
    errors = _build_error_series(real, sitl)
    phase = (
        real.get("phase", pd.Series("other", index=real.index))
        .astype("string")
        .fillna("other")
        .str.lower()
    )

    phase_metrics: dict[str, dict[str, float | None]] = {}
    for phase_name in PHASES:
        mask = phase.eq(phase_name)
        phase_metrics[phase_name] = _phase_metrics(mask, errors, real, sitl, mission, mot_pwm_max)

    overall_mask = pd.Series(True, index=real.index)
    overall = _phase_metrics(overall_mask, errors, real, sitl, mission, mot_pwm_max)

    lag_summary = _lag_estimate_ms(errors.real_speed_mps, errors.sitl_speed_mps)
    osc = _oscillation_score(errors.attitude_error_deg, fs_hz=_infer_hz(real))
    mission_summary = _mission_metrics(real, sitl)
    informational = _informational_metrics(real, sitl)
    units = _units_map()

    overall["lag_median_ms"] = lag_summary["median_ms"]
    overall["lag_iqr_ms"] = lag_summary["iqr_ms"]
    overall["oscillation_peak_hz"] = osc["peak_frequency_hz"]
    overall["oscillation_power"] = osc["integrated_power"]

    return {
        "units": units,
        "alignment": alignment_metadata or {},
        "overall": overall,
        "phases": phase_metrics,
        "mission": mission_summary,
        "informational": informational,
    }


def _phase_metrics(
    mask: pd.Series,
    errors: ErrorSeries,
    real: pd.DataFrame,
    sitl: pd.DataFrame,
    mission: dict[str, Any] | None,
    mot_pwm_max: int,
) -> dict[str, float | None]:
    idx = mask[mask].index
    if len(idx) == 0:
        return {
            "rms_horizontal_error_m": None,
            "rms_altitude_error_m": None,
            "max_horizontal_error_m": None,
            "max_altitude_error_m": None,
            "cross_track_error_m": None,
            "rms_velocity_error_mps": None,
            "overshoot_m": None,
            "motor_saturation_pct": None,
        }

    h = errors.horizontal_error_m.loc[idx]
    alt = errors.altitude_error_m.loc[idx].abs()
    vel = errors.velocity_error_mps.loc[idx]
    cross_track = _cross_track_error(errors.x_sitl_m.loc[idx], errors.y_sitl_m.loc[idx], mission, real)
    overshoot = _overshoot_after_waypoint_transition(h, real.loc[idx, "t_mission_s"], real.loc[idx])
    saturation = _motor_saturation_pct(sitl.loc[idx], mot_pwm_max)

    return {
        "rms_horizontal_error_m": _rms(h),
        "rms_altitude_error_m": _rms(alt),
        "max_horizontal_error_m": _safe_max(h),
        "max_altitude_error_m": _safe_max(alt),
        "cross_track_error_m": cross_track,
        "rms_velocity_error_mps": _rms(vel),
        "overshoot_m": overshoot,
        "motor_saturation_pct": saturation,
    }


def _build_error_series(real: pd.DataFrame, sitl: pd.DataFrame) -> ErrorSeries:
    lat_ref = float(pd.to_numeric(real["lat"], errors="coerce").dropna().iloc[0])
    lon_ref = float(pd.to_numeric(real["lon"], errors="coerce").dropna().iloc[0])
    x_real, y_real = _latlon_to_enu_m(real["lat"], real["lon"], lat_ref, lon_ref)
    x_sitl, y_sitl = _latlon_to_enu_m(sitl["lat"], sitl["lon"], lat_ref, lon_ref)

    horizontal_error_m = np.sqrt((x_real - x_sitl) ** 2 + (y_real - y_sitl) ** 2)
    altitude_error_m = pd.to_numeric(real["alt_m"], errors="coerce") - pd.to_numeric(
        sitl["alt_m"], errors="coerce"
    )

    v_real = pd.concat(
        [
            pd.to_numeric(real.get("vx"), errors="coerce"),
            pd.to_numeric(real.get("vy"), errors="coerce"),
            pd.to_numeric(real.get("vz"), errors="coerce"),
        ],
        axis=1,
    )
    v_sitl = pd.concat(
        [
            pd.to_numeric(sitl.get("vx"), errors="coerce"),
            pd.to_numeric(sitl.get("vy"), errors="coerce"),
            pd.to_numeric(sitl.get("vz"), errors="coerce"),
        ],
        axis=1,
    )
    velocity_error_mps = np.sqrt(((v_real - v_sitl) ** 2).sum(axis=1))
    real_speed = np.sqrt((v_real**2).sum(axis=1))
    sitl_speed = np.sqrt((v_sitl**2).sum(axis=1))

    attitude_error = np.sqrt(
        (
            _angle_diff_deg(real.get("roll"), sitl.get("roll")) ** 2
            + _angle_diff_deg(real.get("pitch"), sitl.get("pitch")) ** 2
            + _angle_diff_deg(real.get("yaw"), sitl.get("yaw")) ** 2
        )
    )

    return ErrorSeries(
        horizontal_error_m=pd.Series(horizontal_error_m, index=real.index),
        altitude_error_m=pd.Series(altitude_error_m, index=real.index),
        velocity_error_mps=pd.Series(velocity_error_mps, index=real.index),
        attitude_error_deg=pd.Series(attitude_error, index=real.index),
        real_speed_mps=pd.Series(real_speed, index=real.index),
        sitl_speed_mps=pd.Series(sitl_speed, index=real.index),
        x_real_m=pd.Series(x_real, index=real.index),
        y_real_m=pd.Series(y_real, index=real.index),
        x_sitl_m=pd.Series(x_sitl, index=real.index),
        y_sitl_m=pd.Series(y_sitl, index=real.index),
    )


def _cross_track_error(
    x: pd.Series,
    y: pd.Series,
    mission: dict[str, Any] | None,
    real: pd.DataFrame,
) -> float | None:
    if not mission:
        return None
    waypoints = [
        wp
        for wp in mission.get("waypoints", [])
        if wp.get("command_name") in {"NAV_WAYPOINT", "NAV_TAKEOFF", "NAV_LAND"}
        and wp.get("lat") is not None
        and wp.get("lon") is not None
    ]
    if len(waypoints) < 2:
        return None

    lat0 = float(waypoints[0]["lat"])
    lon0 = float(waypoints[0]["lon"])
    wp_x, wp_y = _latlon_to_enu_m(
        pd.Series([wp["lat"] for wp in waypoints]),
        pd.Series([wp["lon"] for wp in waypoints]),
        lat0,
        lon0,
    )
    segments = list(zip(zip(wp_x[:-1], wp_y[:-1]), zip(wp_x[1:], wp_y[1:])))
    if not segments:
        return None

    points = np.column_stack([x.to_numpy(dtype=float), y.to_numpy(dtype=float)])
    dists = []
    for px, py in points:
        d = min(_point_to_segment_distance(px, py, a, b) for a, b in segments)
        dists.append(d)
    return float(np.sqrt(np.mean(np.square(dists)))) if dists else None


def _overshoot_after_waypoint_transition(
    horizontal_error_m: pd.Series,
    t_mission_s: pd.Series,
    phase_frame: pd.DataFrame,
) -> float | None:
    seq = pd.to_numeric(phase_frame.get("mission_item_reached_seq"), errors="coerce")
    event_times = pd.to_numeric(t_mission_s[seq.notna()], errors="coerce").to_numpy(dtype=float)
    if event_times.size == 0:
        return None
    overshoots: list[float] = []
    t_arr = pd.to_numeric(t_mission_s, errors="coerce").to_numpy(dtype=float)
    h_arr = pd.to_numeric(horizontal_error_m, errors="coerce").to_numpy(dtype=float)
    for t0 in event_times:
        in_window = (t_arr >= t0) & (t_arr <= t0 + 5.0)
        if not in_window.any():
            continue
        overshoots.append(float(np.nanmax(h_arr[in_window])))
    if not overshoots:
        return None
    return float(np.nanmax(overshoots))


def _lag_estimate_ms(real_speed: pd.Series, sitl_speed: pd.Series) -> dict[str, float | None]:
    a = pd.to_numeric(real_speed, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(sitl_speed, errors="coerce").to_numpy(dtype=float)
    n = min(len(a), len(b))
    if n < 20:
        return {"median_ms": None, "iqr_ms": None}

    hz = 10.0
    win = int(30 * hz)
    if n < win:
        win = n
    step = max(win // 2, 1)
    lags_ms: list[float] = []
    for start in range(0, n - win + 1, step):
        x = a[start : start + win]
        y = b[start : start + win]
        if np.allclose(np.nanstd(x), 0.0) or np.allclose(np.nanstd(y), 0.0):
            continue
        x = np.nan_to_num(x - np.nanmean(x))
        y = np.nan_to_num(y - np.nanmean(y))
        corr = np.correlate(x, y, mode="full")
        lag_idx = int(np.argmax(corr) - (len(x) - 1))
        lags_ms.append(lag_idx * 1000.0 / hz)
    if not lags_ms:
        return {"median_ms": None, "iqr_ms": None}
    arr = np.array(lags_ms)
    return {
        "median_ms": float(np.median(arr)),
        "iqr_ms": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


def _oscillation_score(attitude_error_deg: pd.Series, fs_hz: float) -> dict[str, float | None]:
    y = pd.to_numeric(attitude_error_deg, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(y) < 8:
        return {"peak_frequency_hz": None, "integrated_power": None}
    freqs, pxx = signal.welch(y, fs=fs_hz, nperseg=min(len(y), 256))
    band = (freqs >= 0.5) & (freqs <= 10.0)
    if not np.any(band):
        return {"peak_frequency_hz": None, "integrated_power": None}
    f_band = freqs[band]
    p_band = pxx[band]
    return {
        "peak_frequency_hz": float(f_band[np.argmax(p_band)]),
        "integrated_power": float(np.trapz(p_band, f_band)),
    }


def _motor_saturation_pct(frame: pd.DataFrame, mot_pwm_max: int) -> float | None:
    channels = [f"rcout_{i}" for i in range(1, 9) if f"rcout_{i}" in frame.columns]
    if not channels:
        return None
    values = frame[channels].apply(pd.to_numeric, errors="coerce")
    if values.isna().all().all():
        return None
    sat = (values >= (0.98 * mot_pwm_max)).any(axis=1)
    return float(sat.mean() * 100.0)


def _mission_metrics(real: pd.DataFrame, sitl: pd.DataFrame) -> dict[str, Any]:
    t_real = pd.to_numeric(real["t_mission_s"], errors="coerce")
    t_sitl = pd.to_numeric(sitl["t_mission_s"], errors="coerce")
    flight_time_error = abs(float(t_real.max()) - float(t_sitl.max()))

    mode = sitl.get("mode", pd.Series("", index=sitl.index)).astype("string").str.upper()
    armed = sitl.get("armed", pd.Series(False, index=sitl.index)).fillna(False).astype(bool)
    if mode.str.contains("FAILSAFE", regex=False).any():
        status = "failsafe"
    elif not armed.empty and bool(armed.iloc[-1]) and not mode.isin(["LAND", "RTL"]).any():
        status = "aborted"
    elif mode.isin(["LAND", "RTL"]).any() or (not armed.empty and not bool(armed.iloc[-1])):
        status = "completed"
    else:
        status = "partial"

    return {
        "total_flight_time_error_s": flight_time_error,
        "route_completion_status": status,
    }


def _informational_metrics(real: pd.DataFrame, sitl: pd.DataFrame) -> dict[str, float | None]:
    real_v = pd.to_numeric(real.get("batt_v"), errors="coerce")
    sitl_v = pd.to_numeric(sitl.get("batt_v"), errors="coerce")
    real_a = pd.to_numeric(real.get("batt_a"), errors="coerce")
    sitl_a = pd.to_numeric(sitl.get("batt_a"), errors="coerce")
    return {
        "battery_voltage_error_rms_v": _rms(real_v - sitl_v),
        "battery_current_error_rms_a": _rms(real_a - sitl_a),
    }


def _latlon_to_enu_m(
    lat: pd.Series,
    lon: pd.Series,
    lat_ref: float,
    lon_ref: float,
) -> tuple[pd.Series, pd.Series]:
    lat = pd.to_numeric(lat, errors="coerce")
    lon = pd.to_numeric(lon, errors="coerce")
    r_earth = 6_378_137.0
    lat_ref_rad = np.deg2rad(lat_ref)
    x = np.deg2rad(lon - lon_ref) * r_earth * np.cos(lat_ref_rad)
    y = np.deg2rad(lat - lat_ref) * r_earth
    return pd.Series(x), pd.Series(y)


def _point_to_segment_distance(
    px: float,
    py: float,
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    ax, ay = a
    bx, by = b
    abx = bx - ax
    aby = by - ay
    denom = (abx * abx) + (aby * aby)
    if denom == 0:
        return float(np.hypot(px - ax, py - ay))
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return float(np.hypot(px - proj_x, py - proj_y))


def _angle_diff_deg(a: pd.Series | None, b: pd.Series | None) -> pd.Series:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    return ((x - y + 180.0) % 360.0) - 180.0


def _rms(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(vals))))


def _safe_max(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return None
    return float(np.max(vals))


def _infer_hz(frame: pd.DataFrame) -> float:
    t = pd.to_numeric(frame.get("t_mission_s"), errors="coerce").dropna().to_numpy(dtype=float)
    if len(t) < 2:
        return 10.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 10.0
    return float(1.0 / np.median(dt))


def _units_map() -> dict[str, str]:
    return {
        "rms_horizontal_error_m": "m",
        "rms_altitude_error_m": "m",
        "max_horizontal_error_m": "m",
        "max_altitude_error_m": "m",
        "cross_track_error_m": "m",
        "rms_velocity_error_mps": "m/s",
        "lag_median_ms": "ms",
        "lag_iqr_ms": "ms",
        "overshoot_m": "m",
        "oscillation_peak_hz": "Hz",
        "oscillation_power": "dimensionless",
        "motor_saturation_pct": "%",
        "total_flight_time_error_s": "s",
        "battery_voltage_error_rms_v": "V",
        "battery_current_error_rms_a": "A",
    }
