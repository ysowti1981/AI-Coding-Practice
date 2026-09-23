from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections.abc import AsyncIterator

from ingestor import SpanIngestor
from sampler import TailSampler
from writer import BatchWriter, SpanWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Mock data generation ────────────────────────────────────────────


def _make_trace(tenant_id: str, error: bool = False, slow: bool = False) -> list[dict]:
    """Generate a realistic mock trace (orchestrator → LLM → tool)."""
    trace_id = uuid.uuid4().hex[:32]
    base_ns = time.time_ns()

    duration_ns = int(6e9) if slow else int(random.uniform(0.1, 2.0) * 1e9)

    orch_sid = uuid.uuid4().hex[:16]
    llm_sid = uuid.uuid4().hex[:16]
    tool_sid = uuid.uuid4().hex[:16]

    status = "ERROR" if error else "OK"
    error_attrs = {"error.message": "Tool execution failed: timeout"} if error else {}

    return [
        {
            "trace_id": trace_id,
            "span_id": orch_sid,
            "parent_span_id": None,
            "name": "orchestrator.run",
            "start_time": base_ns,
            "end_time": base_ns + duration_ns,
            "status": status,
            "attributes": {},
            "resource": {"service.name": "agent-orchestrator", "tenant_id": tenant_id},
            "events": [],
        },
        {
            "trace_id": trace_id,
            "span_id": llm_sid,
            "parent_span_id": orch_sid,
            "name": "llm.complete",
            "start_time": base_ns + int(0.05e9),
            "end_time": base_ns + int(duration_ns * 0.4),
            "status": "OK",
            "attributes": {
                "llm.model": "claude-sonnet-4-20250514",
                "llm.input_tokens": str(random.randint(500, 3000)),
                "llm.output_tokens": str(random.randint(100, 1500)),
                "llm.cost_usd": str(round(random.uniform(0.001, 0.05), 4)),
            },
            "resource": {"service.name": "agent-orchestrator", "tenant_id": tenant_id},
            "events": [],
        },
        {
            "trace_id": trace_id,
            "span_id": tool_sid,
            "parent_span_id": orch_sid,
            "name": "tool.execute",
            "start_time": base_ns + int(duration_ns * 0.5),
            "end_time": base_ns + int(duration_ns * 0.95),
            "status": status,
            "attributes": {
                "tool.name": random.choice(
                    ["salesforce_query", "jira_create", "slack_notify", "github_pr"]
                ),
                "http.status_code": "500" if error else "200",
                **error_attrs,
            },
            "resource": {"service.name": "agent-orchestrator", "tenant_id": tenant_id},
            "events": (
                [{"timestamp": base_ns + int(duration_ns * 0.6), "name": "retry_attempted"}]
                if error
                else []
            ),
        },
    ]


async def mock_kafka_consumer(
    num_batches: int = 20,
    spans_per_batch: int = 15,
    interval_s: float = 0.2,
) -> AsyncIterator[list[dict]]:
    """Yield batches of raw span dicts, simulating a Kafka consumer poll loop."""
    tenants = [f"tenant-{i}" for i in range(1, 6)]

    for _ in range(num_batches):
        batch: list[dict] = []
        # ~5 traces per batch (3 spans each)
        for _ in range(spans_per_batch // 3):
            tenant = random.choice(tenants)
            error = random.random() < 0.15
            slow = random.random() < 0.10
            batch.extend(_make_trace(tenant, error=error, slow=slow))
        yield batch
        await asyncio.sleep(interval_s)


# ── Pipeline ────────────────────────────────────────────────────────


async def run_pipeline() -> SpanWriter:
    """Wire all components together and run the ingestion pipeline."""
    # Components
    writer = SpanWriter()
    batch_writer = BatchWriter(
        writer, batch_size=50, flush_interval_s=0.5, max_queue_size=500
    )
    sampler = TailSampler(
        completeness_timeout_s=2.0, sample_rate=0.1, max_buffered_traces=5000
    )
    ingestor = SpanIngestor()

    await batch_writer.start()

    total_ingested = 0
    total_rejected = 0
    total_sampled = 0

    # Main loop: pull from Kafka → validate/transform → sample → batch write
    async for raw_batch in mock_kafka_consumer():
        accepted, rejected = ingestor.ingest(raw_batch)
        total_ingested += len(accepted)
        total_rejected += len(rejected)

        for span in accepted:
            sampler.add_span(span)

        # Emit any traces that have completed (timed out in sampler)
        for trace_spans in sampler.flush_completed():
            total_sampled += len(trace_spans)
            for span in trace_spans:
                await batch_writer.add(span)  # backpressure if queue full

    # Graceful shutdown: force-flush remaining buffered traces
    logger.info("Kafka exhausted — flushing remaining traces from sampler")
    for trace_spans in sampler.flush_all():
        total_sampled += len(trace_spans)
        for span in trace_spans:
            await batch_writer.add(span)

    await batch_writer.stop()

    logger.info("=== Pipeline Summary ===")
    logger.info("  Ingested:  %d spans", total_ingested)
    logger.info("  Rejected:  %d spans", total_rejected)
    logger.info("  Sampled:   %d spans (kept after tail sampling)", total_sampled)
    logger.info("  Written:   %d spans in %d batches", writer.total_written, len(writer.batches))
    logger.info("  Remaining: %d buffered traces", sampler.buffered_trace_count)

    return writer


if __name__ == "__main__":
    asyncio.run(run_pipeline())
