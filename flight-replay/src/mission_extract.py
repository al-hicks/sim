from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .log_extract import iter_dataflash_messages

SUPPORTED_NAV_COMMANDS = {
    16: "NAV_WAYPOINT",
    22: "NAV_TAKEOFF",
    21: "NAV_LAND",
    20: "NAV_RTL",
    178: "DO_CHANGE_SPEED",
}


def extract_mission(bin_path: str | Path) -> dict[str, Any]:
    path = Path(bin_path)
    commands: list[dict[str, Any]] = []
    mode_transitions: list[dict[str, Any]] = []

    for msg_type, t_us, msg in iter_dataflash_messages(path):
        fields = _to_dict(msg)
        if msg_type in {"CMD", "MISSION", "MISSION_CMD"}:
            command_id = _pick_field(fields, "Command", "CNum", "Cmd", "CId")
            if command_id is None:
                continue
            cmd_id = int(command_id)
            command = {
                "t_us": t_us,
                "command_id": cmd_id,
                "command_name": SUPPORTED_NAV_COMMANDS.get(cmd_id, f"UNSUPPORTED_{cmd_id}"),
                "lat": _pick_field(fields, "Lat", "lat"),
                "lon": _pick_field(fields, "Lng", "Lon", "lon"),
                "alt_m": _pick_field(fields, "Alt", "alt"),
                "p1": _pick_field(fields, "P1", "Param1"),
                "p2": _pick_field(fields, "P2", "Param2"),
                "p3": _pick_field(fields, "P3", "Param3"),
                "p4": _pick_field(fields, "P4", "Param4"),
            }
            commands.append(command)
        elif msg_type == "MODE":
            mode = _pick_field(fields, "Mode", "mode")
            if mode is not None:
                mode_transitions.append({"t_us": t_us, "mode": str(mode)})

    return {
        "source_log": str(path),
        "waypoints": commands,
        "mode_transitions": mode_transitions,
        "supported_nav_commands": sorted(SUPPORTED_NAV_COMMANDS.values()),
    }


def write_mission_json(bin_path: str | Path, out_path: str | Path) -> Path:
    mission = extract_mission(bin_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(mission, handle, indent=2)
    return out


def _to_dict(msg: Any) -> dict[str, Any]:
    if hasattr(msg, "to_dict"):
        data = msg.to_dict()
        if isinstance(data, dict):
            return data
    return {}


def _pick_field(fields: dict[str, Any], *candidates: str) -> Any:
    for c in candidates:
        if c in fields and fields[c] is not None:
            return fields[c]
    return None
