from __future__ import annotations

import numpy as np
import pandas as pd

from src.align import align_streams, discover_anchor


def test_discover_anchor_prefers_mission_item_reached():
    df = pd.DataFrame(
        {
            "t_us": [0, 1_000_000, 2_000_000],
            "armed": [True, True, True],
            "mode": ["GUIDED", "AUTO", "AUTO"],
            "mission_item_reached_seq": [np.nan, 1.0, np.nan],
        }
    )
    anchor = discover_anchor(df)
    assert anchor is not None
    assert anchor.anchor_type == "mission_item_reached"
    assert anchor.anchor_t_us == 1_000_000


def test_discover_anchor_fallback_auto_mode_entry():
    df = pd.DataFrame(
        {
            "t_us": [0, 1_000_000, 2_000_000],
            "armed": [True, True, True],
            "mode": ["GUIDED", "AUTO", "AUTO"],
        }
    )
    anchor = discover_anchor(df)
    assert anchor is not None
    assert anchor.anchor_type == "auto_mode_entry"
    assert anchor.anchor_t_us == 1_000_000


def test_align_streams_resamples_to_common_grid():
    real = pd.DataFrame(
        {
            "t_us": [0, 500_000, 1_000_000, 1_500_000],
            "armed": [True, True, True, True],
            "mode": ["AUTO"] * 4,
            "mission_item_reached_seq": [1.0, np.nan, np.nan, np.nan],
            "lat": [1.0, 1.1, 1.2, 1.3],
            "lon": [2.0, 2.1, 2.2, 2.3],
            "alt_m": [10.0, 10.5, 11.0, 11.5],
            "vx": [1.0, 1.0, 1.0, 1.0],
            "vy": [0.0, 0.0, 0.0, 0.0],
            "vz": [0.0, 0.0, 0.0, 0.0],
            "phase": ["cruise"] * 4,
        }
    )
    sitl = real.copy()
    sitl["t_us"] = sitl["t_us"] + 100_000

    aligned = align_streams(real, sitl, resample_hz=10)
    assert len(aligned.real) == len(aligned.sitl)
    diffs = aligned.real["t_mission_s"].diff().dropna().round(3).unique()
    assert len(diffs) == 1
    assert float(diffs[0]) == 0.1
