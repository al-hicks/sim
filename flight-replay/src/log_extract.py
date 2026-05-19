from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from pymavlink import DFReader

from .phases import tag_flight_phases

BASE_COLUMNS = [
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
    "mission_item_reached_seq",
    "phase",
]
RCOUT_COLUMNS = [f"rcout_{i}" for i in range(1, 9)]


def iter_dataflash_messages(path: str | Path) -> Iterator[tuple[str, int, Any]]:
    reader = _open_reader(Path(path))
    while True:
        try:
            msg = reader.recv_msg()
        except Exception:
            break
        if msg is None:
            break
        msg_type = str(getattr(msg, "get_type", lambda: "UNKNOWN")())
        t_us = _message_timestamp_us(msg)
        if t_us is None:
            continue
        yield msg_type, t_us, msg


def extract_log_to_parquet(bin_path: str | Path, out_path: str | Path) -> tuple[Path, dict[str, Any]]:
    frame, meta = extract_log_dataframe(bin_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    meta_path = out.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return out, meta


def extract_log_dataframe(bin_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(bin_path)
    records: list[dict[str, Any]] = []
    version = "unknown"

    for msg_type, t_us, msg in iter_dataflash_messages(path):
        fields = _msg_dict(msg)
        row: dict[str, Any] = {"t_us": int(t_us)}

        if msg_type == "GPS":
            lat = _field(fields, "Lat")
            lon = _field(fields, "Lng", "Lon")
            alt = _field(fields, "Alt")
            if lat is not None:
                row["lat"] = float(lat) / 1e7 if abs(float(lat)) > 1e3 else float(lat)
            if lon is not None:
                row["lon"] = float(lon) / 1e7 if abs(float(lon)) > 1e3 else float(lon)
            if alt is not None:
                row["alt_m"] = float(alt) / 1000.0 if abs(float(alt)) > 1e4 else float(alt)
            spd = _field(fields, "Spd")
            course = _field(fields, "GCrs", "Yaw")
            if spd is not None and course is not None:
                course_rad = np.deg2rad(float(course))
                row["vx"] = float(spd) * float(np.cos(course_rad))
                row["vy"] = float(spd) * float(np.sin(course_rad))

        if msg_type == "ATT":
            row["roll"] = _float_or_nan(_field(fields, "Roll"))
            row["pitch"] = _float_or_nan(_field(fields, "Pitch"))
            row["yaw"] = _float_or_nan(_field(fields, "Yaw"))

        if msg_type in {"CTUN", "XKF1", "NKF1"}:
            vx = _field(fields, "VelX", "VX", "VN")
            vy = _field(fields, "VelY", "VY", "VE")
            vz = _field(fields, "VelZ", "VZ", "VD", "DCRt")
            alt = _field(fields, "Alt", "DAlt", "HAGL")
            if vx is not None:
                row["vx"] = float(vx)
            if vy is not None:
                row["vy"] = float(vy)
            if vz is not None:
                row["vz"] = float(vz)
            if alt is not None and "alt_m" not in row:
                row["alt_m"] = float(alt)

        if msg_type in {"BAT", "BAT1", "BATT"}:
            row["batt_v"] = _float_or_nan(_field(fields, "Volt", "V"))
            row["batt_a"] = _float_or_nan(_field(fields, "Curr", "A"))

        if msg_type in {"RCOU", "RCOUT"}:
            for i in range(1, 9):
                val = _field(fields, f"C{i}", f"Ch{i}")
                if val is not None:
                    row[f"rcout_{i}"] = float(val)

        if msg_type == "MODE":
            mode = _field(fields, "Mode", "mode")
            if mode is not None:
                row["mode"] = str(mode)

        if msg_type in {"ARM", "EV"}:
            armed = _field(fields, "ArmState", "Armed")
            if armed is not None:
                row["armed"] = bool(int(armed))

        if msg_type in {"EKF", "XKF4", "NKF4"}:
            flags = _field(fields, "Flags", "FS")
            if flags is not None:
                row["ekf_flags"] = int(flags)

        if msg_type in {"MISSION_ITEM_REACHED", "MISR", "MISR"}:
            seq = _field(fields, "seq", "Seq", "WP")
            if seq is not None:
                row["mission_item_reached_seq"] = float(seq)

        if msg_type == "MSG":
            text = _field(fields, "Message", "Text")
            if text and "Ardu" in str(text):
                version = str(text)
        if msg_type == "VER" and version == "unknown":
            ver = _field(fields, "FWS", "Version", "Git")
            if ver:
                version = str(ver)

        if len(row) > 1:
            records.append(row)

    if not records:
        frame = _empty_frame()
    else:
        frame = pd.DataFrame.from_records(records)
        frame = frame.sort_values("t_us").drop_duplicates("t_us", keep="last")
        frame = frame.ffill()
        frame["t_mission_s"] = (frame["t_us"] - frame["t_us"].iloc[0]) / 1e6
        for col in BASE_COLUMNS + RCOUT_COLUMNS:
            if col not in frame.columns:
                frame[col] = np.nan
        if "mode" not in frame:
            frame["mode"] = "UNKNOWN"
        frame["armed"] = frame.get("armed", False).fillna(False).astype(bool)
        frame["phase"] = tag_flight_phases(frame)
        frame = frame[BASE_COLUMNS + RCOUT_COLUMNS]

    meta = {
        "source": str(path),
        "ardupilot_version": version,
        "rows": int(len(frame)),
    }
    return frame, meta


def _open_reader(path: Path):
    if hasattr(DFReader, "DFReader_binary"):
        try:
            return DFReader.DFReader_binary(str(path))
        except Exception:
            pass
    if hasattr(DFReader, "DFReader_auto"):
        return DFReader.DFReader_auto(str(path))
    return DFReader.DFReader(str(path))


def _message_timestamp_us(msg: Any) -> int | None:
    fields = _msg_dict(msg)
    for key in ("TimeUS", "time_us", "TimeMs", "TimeMS", "T", "TSec"):
        if key in fields and fields[key] is not None:
            val = float(fields[key])
            if key in {"TimeMs", "TimeMS"}:
                val *= 1000.0
            if key == "TSec":
                val *= 1e6
            return int(val)
    return None


def _msg_dict(msg: Any) -> dict[str, Any]:
    if hasattr(msg, "to_dict"):
        data = msg.to_dict()
        if isinstance(data, dict):
            return data
    return {}


def _field(fields: dict[str, Any], *keys: str):
    for key in keys:
        if key in fields and fields[key] is not None:
            return fields[key]
    return None


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=BASE_COLUMNS + RCOUT_COLUMNS)
    frame["armed"] = frame.get("armed", pd.Series(dtype=bool)).astype(bool)
    return frame
