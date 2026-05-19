from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import log_extract


class _FakeMsg:
    def __init__(self, msg_type: str, **fields):
        self._msg_type = msg_type
        self._fields = fields

    def get_type(self) -> str:
        return self._msg_type

    def to_dict(self):
        return self._fields


def test_extract_log_dataframe_schema(monkeypatch):
    msgs = [
        ("GPS", 1_000_000, _FakeMsg("GPS", TimeUS=1_000_000, Lat=374275000, Lng=-1221697000, Alt=15000, Spd=4.0, GCrs=90)),
        ("ATT", 1_000_000, _FakeMsg("ATT", TimeUS=1_000_000, Roll=1.0, Pitch=2.0, Yaw=3.0)),
        ("CTUN", 1_000_000, _FakeMsg("CTUN", TimeUS=1_000_000, VelX=1.0, VelY=2.0, VelZ=-0.4)),
        ("BAT", 1_000_000, _FakeMsg("BAT", TimeUS=1_000_000, Volt=16.1, Curr=9.3)),
        ("RCOU", 1_000_000, _FakeMsg("RCOU", TimeUS=1_000_000, C1=1200, C2=1250, C3=1300, C4=1350, C5=1400, C6=1450, C7=1500, C8=1550)),
        ("MODE", 1_000_000, _FakeMsg("MODE", TimeUS=1_000_000, Mode="AUTO")),
        ("ARM", 1_000_000, _FakeMsg("ARM", TimeUS=1_000_000, Armed=1)),
        ("MISSION_ITEM_REACHED", 1_000_000, _FakeMsg("MISSION_ITEM_REACHED", TimeUS=1_000_000, seq=1)),
        ("MSG", 1_000_000, _FakeMsg("MSG", TimeUS=1_000_000, Message="ArduCopter V4.5.0")),
    ]

    monkeypatch.setattr(log_extract, "iter_dataflash_messages", lambda _: iter(msgs))
    frame, meta = log_extract.extract_log_dataframe(Path("dummy.BIN"))

    expected = {
        "t_us",
        "t_mission_s",
        "lat",
        "lon",
        "alt_m",
        "vx",
        "vy",
        "vz",
        "roll",
        "pitch",
        "yaw",
        "batt_v",
        "batt_a",
        "mode",
        "armed",
        "ekf_flags",
        "phase",
    }
    assert expected.issubset(set(frame.columns))
    assert frame["armed"].dtype == bool
    assert meta["ardupilot_version"] == "ArduCopter V4.5.0"
    assert isinstance(frame, pd.DataFrame)
