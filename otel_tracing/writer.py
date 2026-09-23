from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from models import Span

logger = logging.getLogger(__name__)


# ── Writer interface ────────────────────────────────────────────────

class SpanWriterProtocol(Protocol):
    async def write_batch(self, spans: list[Span]) -> None: ...


# ── Mock writer (in-memory) ─────────────────────────────────────────

class SpanWriter:
    """Mock writer that stores batches in memory."""

    def __init__(self) -> None:
        self.batches: list[list[Span]] = []
        self.total_written: int = 0

    async def write_batch(self, spans: list[Span]) -> None:
        self.batches.append(spans)
        self.total_written += len(spans)
        logger.info("Wrote batch of %d spans (total: %d)", len(spans), self.total_written)


# ── Batch writer with backpressure ──────────────────────────────────

class BatchWriter:
    """
    Buffers spans and flushes to the underlying writer in batches.

    Flush triggers:
      - batch_size reached
      - flush_interval_s elapsed (whichever comes first)

    Backpressure: the internal asyncio.Queue has a bounded max size.
    When full, `add()` awaits — which propagates pressure upstream
    through the sampler and Kafka consumer, preventing unbounded memory.
    """

    def __init__(
        self,
        writer: SpanWriterProtocol,
        batch_size: int = 100,
        flush_interval_s: float = 1.0,
        max_queue_size: int = 1000,
    ) -> None:
        self._writer = writer
        self._batch_size = batch_size
        self._flush_interval = flush_interval_s
        self._queue: asyncio.Queue[Span] = asyncio.Queue(maxsize=max_queue_size)
        self._buffer: list[Span] = []
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._drain()

    async def add(self, span: Span) -> None:
        """Add a span. Awaits (backpressure) if the internal queue is full."""
        await self._queue.put(span)

    async def _run(self) -> None:
        while self._running:
            try:
                # Block until at least one span arrives or the flush interval elapses
                span = await asyncio.wait_for(
                    self._queue.get(), timeout=self._flush_interval
                )
                self._buffer.append(span)

                # Drain everything currently available without blocking
                while len(self._buffer) < self._batch_size:
                    try:
                        self._buffer.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                if len(self._buffer) >= self._batch_size:
                    await self._flush()

            except asyncio.TimeoutError:
                # Flush interval elapsed — flush whatever we have
                await self._flush()

    async def _flush(self) -> None:
        if not self._buffer:
            return
        batch, self._buffer = self._buffer, []
        await self._writer.write_batch(batch)

    async def _drain(self) -> None:
        """Drain remaining items from the queue and flush."""
        while not self._queue.empty():
            try:
                self._buffer.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        await self._flush()
