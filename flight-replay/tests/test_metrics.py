from __future__ import annotations

from src.metrics import compute_metrics
from tests.helpers import synthetic_real_sitl_frames


def test_compute_metrics_returns_expected_sections():
    real, sitl = synthetic_real_sitl_frames()
    mission = {
        "waypoints": [
            {"command_name": "NAV_TAKEOFF", "lat": 37.4275, "lon": -122.1697},
            {"command_name": "NAV_WAYPOINT", "lat": 37.4277, "lon": -122.1693},
            {"command_name": "NAV_LAND", "lat": 37.4279, "lon": -122.1690},
        ]
    }
    out = compute_metrics(real, sitl, mission=mission, alignment_metadata={"real_anchor_type": "mission_item_reached"})
    assert "overall" in out
    assert "phases" in out
    assert "mission" in out
    assert "units" in out
    assert out["overall"]["rms_horizontal_error_m"] is not None
    assert out["overall"]["lag_median_ms"] is not None
    assert out["mission"]["route_completion_status"] in {"completed", "partial", "aborted", "failsafe"}
