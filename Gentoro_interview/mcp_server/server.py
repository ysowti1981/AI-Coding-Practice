"""MCP state server — exposes house device tools over stdio."""

import os
import sys

# Ensure project root is on sys.path so `agent.telemetry` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

from agent.telemetry import tracer
from mcp_server.state import get_all_status as _get_all_status
from mcp_server.state import get_door_status as _get_door_status
from mcp_server.state import get_temperature as _get_temperature
from mcp_server.state import set_door_status as _set_door_status
from mcp_server.state import set_temperature as _set_temperature

mcp = FastMCP("HomeStateServer")


@mcp.tool()
def get_temperature(room: str) -> dict:
    """Get the current and target temperature (°C) for a room.

    Args:
        room: Room name, e.g. 'living_room' or 'bedroom'.
    """
    with tracer.start_as_current_span(
        "mcp.get_temperature", attributes={"room": room}
    ) as span:
        result = _get_temperature(room)
        span.set_attribute("temperature.current", result["current"])
        span.set_attribute("temperature.target", result["target"])
        return result


@mcp.tool()
def set_temperature(room: str, value: float) -> dict:
    """Set the target temperature (°C) for a room. Returns updated state.

    Args:
        room: Room name, e.g. 'living_room' or 'bedroom'.
        value: Target temperature in Celsius.
    """
    with tracer.start_as_current_span(
        "mcp.set_temperature", attributes={"room": room, "value": value}
    ) as span:
        result = _set_temperature(room, value)
        span.set_attribute("temperature.current", result["current"])
        span.set_attribute("temperature.target", result["target"])
        return result


@mcp.tool()
def get_door_status(door: str) -> str:
    """Get the status ('open' or 'closed') of a door or garage.

    Args:
        door: Door name, e.g. 'front_door', 'back_door', or 'garage'.
    """
    with tracer.start_as_current_span(
        "mcp.get_door_status", attributes={"door": door}
    ) as span:
        status = _get_door_status(door)
        span.set_attribute("door.status", status)
        return status


@mcp.tool()
def set_door_status(door: str, status: str) -> str:
    """Open or close a door or garage. Returns new status.

    Args:
        door: Door name, e.g. 'front_door', 'back_door', or 'garage'.
        status: 'open' or 'closed'.
    """
    with tracer.start_as_current_span(
        "mcp.set_door_status", attributes={"door": door, "status": status}
    ) as span:
        new_status = _set_door_status(door, status)
        span.set_attribute("door.new_status", new_status)
        return new_status


@mcp.tool()
def get_all_status() -> dict:
    """Get a full snapshot of all house devices (temperatures and doors)."""
    with tracer.start_as_current_span("mcp.get_all_status"):
        return _get_all_status()


if __name__ == "__main__":
    mcp.run(transport="stdio")
