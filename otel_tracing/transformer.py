from __future__ import annotations

from models import Span, SpanStatus

# Map: OTel attribute key → (Span field name, type coercion function)
TIER_2_PROMOTIONS: dict[str, tuple[str, type]] = {
    "tool.name": ("tool_name", str),
    "llm.model": ("llm_model", str),
    "llm.input_tokens": ("llm_input_tokens", int),
    "llm.output_tokens": ("llm_output_tokens", int),
    "llm.cost_usd": ("llm_cost_usd", float),
    "http.status_code": ("http_status_code", int),
    "error.message": ("error_message", str),
}


class SpanTransformer:
    """Transforms validated raw span dicts into typed Span objects,
    promoting tier-2 attributes from the attribute map into fixed fields."""

    def transform(self, raw: dict) -> Span:
        attrs = dict(raw.get("attributes", {}))
        resource = dict(raw.get("resource", {}))

        # Promote tier-2 attributes out of the map into typed fields
        tier2: dict[str, object] = {}
        for attr_key, (field_name, coerce) in TIER_2_PROMOTIONS.items():
            if attr_key in attrs:
                try:
                    tier2[field_name] = coerce(attrs.pop(attr_key))
                except (ValueError, TypeError):
                    pass  # coercion failed — leave in attrs map

        start_ns = int(raw["start_time"])
        end_ns = int(raw["end_time"])
        duration_ms = (end_ns - start_ns) / 1_000_000

        # Pull tenant_id and service_name out of resource (promoted to tier-1)
        tenant_id = resource.pop("tenant_id", None) or attrs.pop("tenant_id", "unknown")
        service_name = resource.pop("service.name", "unknown")

        return Span(
            trace_id=raw["trace_id"],
            span_id=raw["span_id"],
            parent_span_id=raw.get("parent_span_id"),
            name=raw["name"],
            start_time=start_ns,
            end_time=end_ns,
            duration_ms=duration_ms,
            status=SpanStatus(raw["status"]),
            service_name=service_name,
            tenant_id=tenant_id,
            attributes=attrs,
            resource_attributes=resource,
            events=raw.get("events", []),
            **tier2,
        )
