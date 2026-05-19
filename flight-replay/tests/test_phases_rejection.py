from __future__ import annotations

import pandas as pd

from src.config import RejectionConfig
from src.phases import detect_rejection, tag_flight_phases
from tests.helpers import synthetic_real_sitl_frames


def test_phase_tagging_assigns_expected_labels():
    real, _ = synthetic_real_sitl_frames()
    phase = tag_flight_phases(real)
    assert set(phase.unique()).issubset({"takeoff", "cruise", "transition", "loiter", "land", "other"})
    assert (phase == "transition").any()


def test_rejection_logic_flags_short_flight():
    frame = pd.DataFrame(
        {
            "t_us": [0, 1_000_000, 2_000_000],
            "armed": [True, True, False],
            "mode": ["GUIDED", "GUIDED", "GUIDED"],
            "lat": [None, None, None],
            "lon": [None, None, None],
            "alt_m": [None, None, None],
            "roll": [None, None, None],
            "pitch": [None, None, None],
            "yaw": [None, None, None],
            "vx": [None, None, None],
            "vy": [None, None, None],
            "vz": [None, None, None],
        }
    )
    result = detect_rejection(frame, cfg=RejectionConfig(min_armed_seconds=10, max_gap_pct=5), anchor_found=False)
    assert result.rejected
    assert "armed_time_below_minimum" in result.reasons
    assert "alignment_anchor_not_found" in result.reasons
