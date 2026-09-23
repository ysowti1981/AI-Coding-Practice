from __future__ import annotations

REQUIRED_FIELDS = {"trace_id", "span_id", "name", "start_time", "end_time", "status"}
VALID_STATUSES = {"OK", "ERROR", "UNSET"}


class SpanValidator:
    """Validates raw OTel span dicts before transformation."""

    def validate(self, raw: dict) -> list[str]:
        """Return a list of error strings. Empty list means the span is valid."""
        errors: list[str] = []

        for f in REQUIRED_FIELDS:
            if f not in raw:
                errors.append(f"Missing required field: {f}")

        if "status" in raw and raw["status"] not in VALID_STATUSES:
            errors.append(f"Invalid status: {raw['status']}")

        if "start_time" in raw and "end_time" in raw:
            if not isinstance(raw["start_time"], (int, float)):
                errors.append("start_time must be numeric")
            elif not isinstance(raw["end_time"], (int, float)):
                errors.append("end_time must be numeric")
            elif raw["end_time"] < raw["start_time"]:
                errors.append("end_time must be >= start_time")

        resource = raw.get("resource")
        if resource is None:
            errors.append("Missing resource")
        elif "service.name" not in resource:
            errors.append("resource missing service.name")

        return errors
