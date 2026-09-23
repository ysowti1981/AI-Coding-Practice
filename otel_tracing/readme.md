# Round 2 — Alternative Design

**Theme:** Production observability and evaluation pipeline for a live MCP agent platform.

**Why this fits Gentoro:** their job description heavily emphasizes "Analyze telemetry and execution traces to create feedback loops for continuous agent improvement and automated evaluation" and "Build and operate observability stacks (e.g., OpenTelemetry) to monitor agent reasoning paths, tool usage, and performance in real-time." This is half their pitch — observability is one of their three pillars.

**What this round forces you to think about (different from Round 1):**

- Streaming data pipelines — agent traces flow continuously, not as one-off requests
- Trace data modeling — hierarchical span data, high cardinality, time-series storage (Clickhouse is literally in the JD)
- Real-time aggregation + alerting — detecting tool selection failures, latency regressions, cost spikes
- LLM-as-judge for evaluation at scale — running automated quality checks on production traffic
- Sampling strategies — you can't store everything, so what do you keep?
- Closing the loop — turning production failures into eval datasets, then into prompt/model improvements
- Multi-tenant isolation — different customers' data must be cleanly separated

**What this avoids re-covering:** retrieval, agent loops, tool registries — Round 1 already exercised those.

---

## Session Plan (2 hours)

**Phase 1 (0–15 min): Architecture discussion**
The problem: design a system that ingests OpenTelemetry traces from many customer agent deployments, stores them efficiently, runs automated quality evaluations on them, surfaces failures, and feeds insights back into agent improvement. Discuss components before coding.

**Phase 2 (15–55 min): Core build — Trace ingestion + storage**
Build a service that accepts OTel spans, models them as structured data, stores them queryably, and exposes basic queries (find slow tool calls, find failed tool selections, etc.). I'll interrupt around minute 35 to dig into data modeling, cardinality, sampling, schema design.

**Phase 3 (55–90 min): LLM-as-judge evaluation pipeline**
Build a worker that pulls completed traces and runs automated quality scoring — was the tool selection appropriate? Did the agent make reasoning errors? Did it hallucinate? I'll pause around minute 80 to discuss judge model design, calibration, false positive rates, cost.

**Phase 4 (90–115 min): Feedback loop architecture**
Discuss how to take production failures and turn them into eval datasets, prompt improvements, fine-tuning data. Light coding on a "failure → eval case" pipeline.

---

## Phase 1 — Architecture Discussion

### Round 1 — Opening Question

Walk me through the high-level architecture. Specifically I want to hear:

1. What are the major components, end to end, from "trace gets emitted" to "insight gets surfaced"?
2. What does the data flow look like — sync, async, streaming?
3. Where does data live, in what form, and why?
4. Where does the LLM appear in this system?
5. What are the failure modes you're worried about?

#### Expected Answer

**Key architectural decisions:**

**Trace storage — use Clickhouse, not Redis or a relational DB.**

Redis is an in-memory key-value store — used for hot caching and pub-sub, not for storing trace data at scale. You'd lose data on restart unless configured very carefully, and the cost per GB is much higher than disk-based storage.

Traces are highly structured. A span has a defined shape: `trace_id`, `span_id`, `parent_span_id`, `name`, `start_time`, `end_time`, `status`, `attributes` (key-value), `events` (list). Every OpenTelemetry span follows this schema. The attributes are flexible, but the envelope is rigid.

The right choice is a **column-oriented time-series store** — specifically **Clickhouse** (it's literally listed in Gentoro's job description). Honeycomb, Grafana Tempo, Signoz, Uptrace — most modern observability backends are built on Clickhouse or similar columnar stores. Why Clickhouse:

- Columnar storage compresses trace data extremely well (~10–50x ratios are normal)
- Time-series queries (`give me all spans where tool_name='salesforce_query' and latency_ms > 5000 in the last hour`) are native and fast
- Aggregations across billions of rows are interactive (sub-second)
- Handles high-cardinality attributes well (which RDBMSes don't)
- Schema-on-write but with flexible JSON/Map columns for attributes

A relational DB doesn't survive contact with trace volumes. A single agent call can produce 20–100 spans. A medium customer might emit 10M spans/day. You'd be writing at hundreds of thousands of rows per second, joining across many tables to reconstruct a trace, and your indexes would be enormous. Postgres falls over here. Relational storage belongs in this system — but only for **metadata** (customers, agent definitions, eval datasets, dashboards) — not for trace data itself.

**Two processing paths — hot and cold.**

Some processing must be near-real-time (alerting on a tool selection failure rate spike), some can be batch (running expensive LLM-judge evaluations overnight). A senior answer specifies **two paths**: a hot path for real-time aggregation/alerting and a cold path for batch evaluation and analysis. This is the **lambda architecture** pattern, or a streamlined version sometimes called **kappa architecture** when both paths use the same streaming engine.

**Sampling is mandatory at scale.**

At trace volumes, you can't store everything. Two flavors:

- **Head-based sampling** — decide at trace start whether to keep it (e.g., 10% random sample). Cheap, but you might miss the failures.
- **Tail-based sampling** — buffer the whole trace, then decide based on outcome (always keep errored traces, slow traces, traces that triggered HITL, plus a sample of normal ones). More accurate but requires buffering, and the buffer adds complexity.

For evaluation, you want tail-based — you specifically want the weird traces, not random ones.

**Multi-tenancy must be addressed from day one.**

How do you isolate Customer A's traces from Customer B's? Tenant ID partitioning at the storage layer? Separate Clickhouse tables per tenant? Row-level security? This is a Gentoro selling point — they sell to enterprises who care a lot about this.

**Failure modes to name explicitly:**

Ingestion lag (Kafka backs up), schema drift (customer changes their span attributes), poison messages (malformed spans crash a worker), LLM-judge cost runaway, evaluation lag making findings stale, customer trace volume causing noisy-neighbor problems for other tenants.

#### The Complete Architecture

**Hot path (real-time):**

Customer agents emit OTel traces → OTel Collector (per region) → Kafka (durable, partitioned by `tenant_id`) → stream processor (Flink, or simpler Kafka consumers) → Clickhouse (trace store) + Redis (hot cache for currently-running traces, for tail sampling buffer). Same stream → real-time aggregator computing rolling metrics (p99 latency per tool, error rate per tenant) → Prometheus or Clickhouse materialized views → alerting.

**Cold path (batch evaluation):**

Scheduled job pulls traces from Clickhouse (e.g., hourly) → eval queue (Kafka or SQS) → LLM judge workers (autoscaled pool) → write eval results back to Clickhouse, alongside the original traces. Failed evals get flagged → eval dataset builder turns them into golden test cases → feeds back into prompt/agent improvement.

**Metadata plane (small data, structured):**

Postgres for: customer accounts, agent configurations, eval rubric definitions, golden datasets, dashboard configs, alert rules.

**Query/serving plane:**

API service for engineers to query traces ("show me trace X") and analytics ("tool selection accuracy by customer over time"). Customer-facing dashboards read from same Clickhouse.

**Key takeaway:** Clickhouse for traces, Kafka for ingestion, two paths (hot for monitoring, cold for evaluation), Postgres for metadata, multi-tenant from day one.

---

### Round 2 — Re-design Question (Narrowed Scope)

Forget aggregation and LLM-judge for a moment. Just focus on the trace ingestion and storage portion of this system. Walk me through:

1. The data flow from customer agent → Clickhouse
2. The Clickhouse schema you'd design for spans (what columns, what's indexed, what's a Map vs a regular column)
3. How you'd partition and shard for multi-tenant isolation and query performance
4. How you'd handle a customer whose agent produces a sudden 100x spike in trace volume

#### Sub-Question: Clickhouse Schema Design

An OTel span is roughly:

- `trace_id` (16 bytes, hex)
- `span_id` (8 bytes, hex)
- `parent_span_id` (nullable)
- `name` (string — e.g., `'orchestrator.run'`, `'llm.complete'`, `'tool.execute'`)
- `start_time` (nanosecond precision timestamp)
- `end_time` (nanosecond precision timestamp)
- `status` (OK / ERROR / UNSET)
- `attributes` — a key-value map. Variable per span. For a tool execution span, attributes might include: `tool.name`, `tool.endpoint`, `tool.tenant_id`, `tool.latency_ms`, `tool.retry_count`, `http.status_code`. For an LLM call span: `llm.model`, `llm.input_tokens`, `llm.output_tokens`, `llm.cost_usd`. For a retrieval span: `retrieval.top_k`, `retrieval.candidate_tools`, `retrieval.scores`.
- `events` — a list of timestamped events within the span (e.g., `'retry attempted'`, `'circuit breaker opened'`)
- `resource` — metadata about where the span came from: `service.name`, `service.version`, `host`, `region`, `deployment.environment`

Design the Clickhouse table. Things to think about:

- Which fields go in fixed columns vs. inside an attributes Map?
- What's your ORDER BY clause? (In Clickhouse, ORDER BY defines the sort key, which determines query performance.)
- What's your PARTITION BY clause? (In Clickhouse, partitions are units of data management.)
- What columns do you put codecs/compression on?
- How do you handle the variable attribute keys without making queries painful?

#### Expected Answer: Three-Tier Column Decomposition

Don't put all attributes in a Map — some attributes are **hot** (queried constantly) and should be promoted to fixed columns even though they "feel" like attributes.

**Tier 1 — Fixed columns (always present, always queried):**

- `trace_id`, `span_id`, `parent_span_id`
- `name`, `start_time`, `end_time`, `duration_ms` (computed)
- `status` (enum)
- `service_name`, `tenant_id` — pulled OUT of resource/attributes and promoted because every single query filters by them

**Tier 2 — Promoted hot attributes (specific known fields you query often):**

- `tool_name` (nullable string) — for spans where `name='tool.execute'`
- `llm_model` (nullable string) — for LLM spans
- `llm_input_tokens`, `llm_output_tokens`, `llm_cost_usd` — for cost analytics
- `http_status_code` — for error analysis
- `error_message` (nullable, low cardinality)

These are nullable because not every span has them, but when they're present, fixed columns crush Map lookups for filter/aggregate queries.

**Tier 3 — Attribute Map (the long tail):**

- `attributes Map(String, String)` — for the genuinely variable stuff
- `resource_attributes Map(String, String)` — for resource metadata

The architectural rule is: **promote any attribute that appears in your top 20 queries**. You can also use materialized columns in Clickhouse that auto-extract from the Map into a typed column at write time (hybrid approach).

#### Expected Answer: Partitioning and Sort Key

The correct pattern for trace data in Clickhouse:

```sql
PARTITION BY toDate(start_time)
ORDER BY (tenant_id, service_name, name, start_time)
```

**Why `PARTITION BY toDate(start_time)` not `tenant_id`:**

- `PARTITION BY tenant_id` doesn't scale — 10,000 customers means 10,000 partitions. Clickhouse recommends keeping partition count manageable (under a few thousand). Each partition has overhead (file handles, merge complexity, metadata).
- Partition by day gives predictable bounded partitions, easy data retention with `ALTER TABLE DROP PARTITION`, and partition pruning makes time-range queries fast.

**Why `ORDER BY (tenant_id, service_name, name, start_time)`:**

- `tenant_id` first because it's the most selective and most common filter
- Then service and span name to cluster similar spans together
- Then `start_time` within the cluster

This means Clickhouse can:

- Skip entire days for time-range queries (partition pruning)
- Skip entire tenants within a day (sort-key skipping)
- Co-locate spans for the same service to compress better
- Resolve "last hour for tenant X" in milliseconds

**Special case for trace lookups by trace_id:** queries like "fetch this whole trace" don't fit the `(tenant, service, name, time)` sort order. Solution: either a secondary skip index on `trace_id`, or a separate denormalized table sorted by `(trace_id, span_id)` — duplicating the data — optimized for trace reconstruction. Storage is cheap, query latency is the customer experience.

---

#### Sub-Question: Traffic Spike Handling

A customer's agent goes haywire — maybe a bug causes it to retry 1000x in a loop. They go from 100 spans/sec to 50,000 spans/sec. This noisy neighbor will:

- Saturate their Kafka partition (head-of-line blocking — their spans queue up)
- Push their writes ahead of other tenants' writes if you share consumer threads
- Bloat your Clickhouse storage costs
- Generate thousands of pages of useless redundant data

What concrete mechanisms protect your system and your other tenants? (At least 4–5 distinct techniques.)

#### Expected Answer: Defense in Depth (8 Mechanisms)

1. **Per-tenant Kafka quotas.** Kafka itself supports producer/consumer quotas (`producer_byte_rate`, `consumer_byte_rate`) per client. Set this so a single tenant can't dominate cluster bandwidth.

2. **Rate limiting at the OTel collector.** Drop spans at the edge before they enter Kafka. Token bucket per tenant. Cheap and immediate.

3. **Tail-based sampling at the collector.** When a tenant goes haywire, dynamically increase their sampling rate so you keep representative traces but discard the firehose. Smart sampling gives you "always keep errors + slow + 1% of normal" even under load.

4. **Per-tenant Kafka partitions / partition isolation.** Big tenants get dedicated partitions; small tenants share. This bounds the blast radius — if one tenant's partition is backed up, others move on. (Technically: keyed partitioning still has noisy-neighbor risk if a hot key dominates a partition; you may need to rebalance.)

5. **Storage quotas + billing limits.** Hard quota: tenant exceeds X spans/day, additional spans are dropped or routed to a cold cheap tier. Often customer-tier-dependent (free tier vs. enterprise).

6. **Circuit breaker at ingestion.** If write latency to Clickhouse exceeds a threshold, shed load — refuse new spans temporarily. Prevents cascading failure.

7. **Anomaly detection on per-tenant volume.** Flag a tenant when their ingestion rate is N standard deviations above their own baseline. Alert their account team and/or auto-throttle.

8. **Span-level deduplication.** If the runaway agent is in a retry loop emitting near-identical spans, deduplicate at ingestion to compress the storm.

**The senior framing:** you defend in depth. Edge limits, partition isolation, smart sampling, storage quotas, circuit breakers — five layers, each catches different failure modes. You don't pick one; you stack them.
