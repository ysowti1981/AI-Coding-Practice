from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SpanStatus(Enum):
    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


@dataclass
class Span:
    # Tier 1 — Fixed columns (always present, always queried)
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time: int  # nanoseconds epoch
    end_time: int  # nanoseconds epoch
    duration_ms: float  # computed: (end - start) / 1e6
    status: SpanStatus
    service_name: str
    tenant_id: str

    # Tier 2 — Promoted hot attributes (nullable, query-critical)
    tool_name: str | None = None
    llm_model: str | None = None
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    llm_cost_usd: float | None = None
    http_status_code: int | None = None
    error_message: str | None = None

    # Tier 3 — Attribute maps (long tail)
    attributes: dict[str, str] = field(default_factory=dict)
    resource_attributes: dict[str, str] = field(default_factory=dict)

    # Events
    events: list[dict] = field(default_factory=list)
