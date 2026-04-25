import time
from typing import Any, Dict, List, Optional


class Span:
    """A single unit of work in a trace, modeled after OpenTelemetry spans."""

    def __init__(self, name: str, parent: Optional["Span"] = None):
        self.name = name
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.attributes: Dict[str, Any] = {}
        self.status: str = "UNSET"
        self.children: List["Span"] = []
        self.parent = parent
        if parent is not None:
            parent.children.append(self)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: str) -> None:
        self.status = status

    def end(self) -> None:
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round((self.end_time - self.start_time) * 1000, 2)
            if self.end_time
            else None,
            "attributes": self.attributes,
            "status": self.status,
            "children": [child.to_dict() for child in self.children],
        }


class Tracer:
    """Lightweight OpenTelemetry-style tracer that collects spans as nested dicts."""

    def __init__(self):
        self._root_spans: List[Span] = []
        self._current_span: Optional[Span] = None

    def start_span(self, name: str) -> Span:
        span = Span(name, parent=self._current_span)
        if self._current_span is None:
            self._root_spans.append(span)
        self._current_span = span
        return span

    def end_span(self, span: Span, status: str = "OK") -> None:
        span.set_status(status)
        span.end()
        self._current_span = span.parent

    def get_trace(self) -> List[Dict[str, Any]]:
        return [span.to_dict() for span in self._root_spans]
