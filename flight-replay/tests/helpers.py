from __future__ import annotations

import numpy as np
import pandas as pd

from src.phases import tag_flight_phases


def synthetic_real_sitl_frames(duration_s: int = 60, hz: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = np.arange(0.0, duration_s, 1.0 / hz)
    t_us = (t * 1_000_000).astype(np.int64) + 5_000_000

    lat0 = 37.4275
    lon0 = -122.1697
    lat = lat0 + (0.00004 * np.sin(0.05 * t))
    lon = lon0 + (0.00035 * (t / duration_s))
    alt = 10.0 + 0.3 * t + 0.4 * np.sin(0.3 * t)
    vx = np.gradient(lon, t, edge_order=2) * 111_320.0 * np.cos(np.deg2rad(lat0))
    vy = np.gradient(lat, t, edge_order=2) * 111_320.0
    vz = np.gradient(alt, t, edge_order=2)

    roll = 2.0 * np.sin(0.8 * t)
    pitch = 1.5 * np.sin(0.6 * t)
    yaw = 45.0 + 4.0 * np.sin(0.2 * t)
    batt_v = 16.8 - (0.01 * t)
    batt_a = 8.0 + 0.5 * np.sin(0.5 * t)

    mode = np.array(["AUTO"] * len(t), dtype=object)
    mode[t > (duration_s - 10)] = "LAND"
    mode[t < 4] = "GUIDED"
    armed = np.array([True] * len(t))
    armed[-1] = False

    reached = np.full(len(t), np.nan)
    reached[int(5 * hz)] = 1
    reached[int(25 * hz)] = 2
    reached[int(45 * hz)] = 3

    real = pd.DataFrame(
        {
            "t_us": t_us,
            "t_mission_s": t,
            "lat": lat,
            "lon": lon,
            "alt_m": alt,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "batt_v": batt_v,
            "batt_a": batt_a,
            "mode": mode,
            "armed": armed,
            "ekf_flags": np.zeros(len(t)),
            "mission_item_reached_seq": reached,
            "rcout_1": 1450 + 200 * np.sin(0.4 * t),
            "rcout_2": 1500 + 220 * np.sin(0.45 * t),
            "rcout_3": 1490 + 300 * np.sin(0.42 * t),
            "rcout_4": 1470 + 250 * np.sin(0.41 * t),
            "rcout_5": 1400 + 100 * np.sin(0.5 * t),
            "rcout_6": 1500 + 100 * np.sin(0.55 * t),
            "rcout_7": 1500 + 80 * np.sin(0.58 * t),
            "rcout_8": 1500 + 90 * np.sin(0.62 * t),
        }
    )

    lag_s = 0.3
    t_shift = np.clip(t - lag_s, t[0], t[-1])
    sitl = pd.DataFrame(
        {
            "t_us": t_us + 40_000,
            "t_mission_s": t,
            "lat": np.interp(t_shift, t, lat) + 0.000002,
            "lon": np.interp(t_shift, t, lon) - 0.000003,
            "alt_m": np.interp(t_shift, t, alt) + 0.25,
            "vx": np.interp(t_shift, t, vx) * 1.03,
            "vy": np.interp(t_shift, t, vy) * 0.97,
            "vz": np.interp(t_shift, t, vz) * 1.05,
            "roll": np.interp(t_shift, t, roll) + 0.8 * np.sin(3.0 * t),
            "pitch": np.interp(t_shift, t, pitch) + 0.7 * np.sin(2.5 * t),
            "yaw": np.interp(t_shift, t, yaw) + 1.2 * np.sin(2.2 * t),
            "batt_v": np.interp(t_shift, t, batt_v) + 0.2,
            "batt_a": np.interp(t_shift, t, batt_a) - 0.3,
            "mode": mode,
            "armed": armed,
            "ekf_flags": np.zeros(len(t)),
            "mission_item_reached_seq": reached,
            "rcout_1": 1500 + 470 * np.sin(0.4 * t),
            "rcout_2": 1500 + 480 * np.sin(0.45 * t),
            "rcout_3": 1500 + 490 * np.sin(0.42 * t),
            "rcout_4": 1500 + 500 * np.sin(0.41 * t),
            "rcout_5": 1500 + 300 * np.sin(0.5 * t),
            "rcout_6": 1500 + 300 * np.sin(0.55 * t),
            "rcout_7": 1500 + 290 * np.sin(0.58 * t),
            "rcout_8": 1500 + 280 * np.sin(0.62 * t),
        }
    )

    real["phase"] = tag_flight_phases(real)
    sitl["phase"] = tag_flight_phases(sitl)
    return real, sitl
