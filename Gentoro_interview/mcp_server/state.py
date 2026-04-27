"""In-memory house state store.

This is the single source of truth for all device state.
Only the MCP server should call these functions — the orchestrator
must go through MCP tools.
"""

from typing import Any

_state: dict[str, Any] = {
    "temperature": {
        "living_room": {"current": 21.5, "target": 22.0},
        "bedroom": {"current": 20.0, "target": 20.0},
    },
    "doors": {
        "front_door": "closed",
        "back_door": "closed",
        "garage": "open",
    },
}

VALID_ROOMS = list(_state["temperature"].keys())
VALID_DOORS = list(_state["doors"].keys())


def get_temperature(room: str) -> dict[str, float]:
    """Return {"current": ..., "target": ...} for a room."""
    if room not in _state["temperature"]:
        raise ValueError(f"Unknown room '{room}'. Valid rooms: {VALID_ROOMS}")
    return dict(_state["temperature"][room])


def set_temperature(room: str, value: float) -> dict[str, float]:
    """Set target temperature for a room. Returns updated state."""
    if room not in _state["temperature"]:
        raise ValueError(f"Unknown room '{room}'. Valid rooms: {VALID_ROOMS}")
    _state["temperature"][room]["target"] = value
    return dict(_state["temperature"][room])


def get_door_status(door: str) -> str:
    """Return 'open' or 'closed' for a door/garage."""
    if door not in _state["doors"]:
        raise ValueError(f"Unknown door '{door}'. Valid doors: {VALID_DOORS}")
    return _state["doors"][door]


def set_door_status(door: str, status: str) -> str:
    """Set door/garage to 'open' or 'closed'. Returns new status."""
    if door not in _state["doors"]:
        raise ValueError(f"Unknown door '{door}'. Valid doors: {VALID_DOORS}")
    if status not in ("open", "closed"):
        raise ValueError(f"Status must be 'open' or 'closed', got '{status}'")
    _state["doors"][door] = status
    return _state["doors"][door]


def get_all_status() -> dict[str, Any]:
    """Return full snapshot of all house devices."""
    return {
        "temperature": {
            room: dict(data) for room, data in _state["temperature"].items()
        },
        "doors": dict(_state["doors"]),
    }
