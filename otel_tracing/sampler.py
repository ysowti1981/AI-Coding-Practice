from __future__ import annotations

import random
import time

from models import Span, SpanStatus


class TailSampler:
    """
    Buffers spans by trace_id. When a trace is "complete" (no new spans
    for `completeness_timeout_s` seconds), applies sampling rules:

      1. Always keep traces containing any ERROR span
      2. Always keep traces whose wall-clock duration exceeds 5 seconds
      3. Otherwise keep with probability `sample_rate`

    Memory is bounded by `max_buffered_traces` — oldest trace is evicted
    when the buffer is full.
    """

    def __init__(
        self,
        completeness_timeout_s: float = 5.0,
        sample_rate: float = 0.1,
        max_buffered_traces: int = 10_000,
    ) -> None:
        self._buffers: dict[str, list[Span]] = {}
        self._last_seen: dict[str, float] = {}
        self._completeness_timeout = completeness_timeout_s
        self._sample_rate = sample_rate
        self._max_buffered_traces = max_buffered_traces

    @property
    def buffered_trace_count(self) -> int:
        return len(self._buffers)

    def add_span(self, span: Span) -> None:
        tid = span.trace_id
        if tid not in self._buffers:
            self._evict_if_full()
            self._buffers[tid] = []
        self._buffers[tid].append(span)
        self._last_seen[tid] = time.monotonic()

    def flush_completed(self) -> list[list[Span]]:
        """Return traces that timed out and passed sampling. Removes them from the buffer."""
        now = time.monotonic()
        expired: list[str] = [
            tid
            for tid, last_seen in self._last_seen.items()
            if now - last_seen >= self._completeness_timeout
        ]

        kept: list[list[Span]] = []
        for tid in expired:
            spans = self._buffers.pop(tid)
            del self._last_seen[tid]
            if self._should_keep(spans):
                kept.append(spans)
        return kept

    def flush_all(self) -> list[list[Span]]:
        """Force-flush every buffered trace through the sampling rules (for graceful shutdown)."""
        kept: list[list[Span]] = []
        for tid in list(self._buffers):
            spans = self._buffers.pop(tid)
            del self._last_seen[tid]
            if self._should_keep(spans):
                kept.append(spans)
        return kept

    def _should_keep(self, spans: list[Span]) -> bool:
        # Rule 1: always keep traces with any errored span
        if any(s.status == SpanStatus.ERROR for s in spans):
            return True

        # Rule 2: always keep traces longer than 5 seconds
        trace_start = min(s.start_time for s in spans)
        trace_end = max(s.end_time for s in spans)
        duration_s = (trace_end - trace_start) / 1_000_000_000
        if duration_s > 5.0:
            return True

        # Rule 3: probabilistic sample
        return random.random() < self._sample_rate

    def _evict_if_full(self) -> None:
        if len(self._buffers) >= self._max_buffered_traces:
            oldest_tid = min(self._last_seen, key=lambda k: self._last_seen[k])
            del self._buffers[oldest_tid]
            del self._last_seen[oldest_tid]
