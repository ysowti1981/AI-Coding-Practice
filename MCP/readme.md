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

**Phase 5 (115–120 min): Wrap-up + your questions for me**
