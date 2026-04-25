# Tool Orchestration Service

## Problem

Build a service that sits between an LLM-based agent and a registry of hundreds of enterprise tools — things like "create a Jira ticket," "query Salesforce contacts," "send a Slack message," etc. Each tool has a name, a natural language description, and a JSON schema for its parameters.

When the agent receives a user request like _"find all open deals over $50k and notify the sales team on Slack,"_ the service needs to:

1. **Identify which tools are relevant** (semantic matching)
2. **Plan the execution order** (e.g., Salesforce query first, then Slack notification)
3. **Execute with proper error handling and observability**

---

## Phase 0 — System Design (Verbal)

You're designing a Tool Resolution and Orchestration Service for an agentic platform. Hundreds of enterprise tools are registered. An LLM agent sends a natural language request, and your service needs to find the right tools, plan execution, run them, and return results — all while being observable and secure.

Walk me through the architecture. I want to hear about:

1. What services or components exist
2. How they communicate
3. Where data lives and in what form
4. What happens when things fail

---

## Phase 1 — Tool Registry (~20 min)

Build a `ToolRegistry` service in Python that does the following:

- Stores tool definitions — each tool has a **name**, **description**, **parameters** (as a JSON schema), and an **endpoint URL**
- When a tool is registered, it **embeds the tool's description** using a sentence embedding model
- Exposes a `search(query: str, top_k: int) -> List[Tool]` method that takes a natural language query, embeds it, and returns the top-k most similar tools by **cosine similarity**

For the embedding model, you can use `sentence-transformers` or even just mock it with random vectors for now — whichever gets you moving faster. I care more about the architecture of the code than the specific model.

### Phase 1 — Interview Questions

**Q1 — Embedding choice:**
You picked `all-MiniLM-L6-v2`. Why that over something like `BAAI/bge-small-en-v1.5`, or OpenAI's `text-embedding-3-small`, or a larger model like `all-mpnet-base-v2`? What's the tradeoff you're making, and how would you decide in a production setting?

**Q2 — What you embed:**
You said `f"{self.name}: {self.description}"`. Tool names are often short and arbitrary — like `sf_deals_query` or `slackNotify`. Could that actually hurt retrieval quality? What would you add or change to improve matching for real-world queries?

**Q3 — Scaling concern:**
Your `register` method does `np.vstack` every time a tool is added (O(n) per insertion because it copies the whole matrix). And your search does `np.argsort` over all embeddings (O(n log n)). At 300 tools this is fine, but at 30,000 or 300,000 it isn't.
- When would this become a problem? Give rough numbers.
- What would you swap in, and what are the tradeoffs?

### Phase 1 — Key Takeaways

<details>
<summary>Q1 — Embedding model selection (click to expand)</summary>

The pragmatic answer ("pick small, evaluate, scale up") is fine but surface-level. Dimensions a senior engineer should explicitly name:

- **Dimensionality vs. speed vs. quality** — MiniLM is 384-dim, mpnet is 768-dim, OpenAI's text-embedding-3-large is 3072-dim. Higher dim = more storage, slower search, better semantic resolution. At 300 tools it doesn't matter; at 300k tools, 3072 vs 384 is an 8x storage/compute difference.
- **MTEB benchmarks** — the standard leaderboard for comparing embedding models on retrieval tasks. Know it exists.
- **Domain fit** — general-purpose models are trained on web data. For tool/API descriptions, a code-aware model (e.g., `jinaai/jina-embeddings-v2-base-code`) might do better.
- **Hosted vs. self-hosted** — OpenAI/Voyage/Cohere APIs = no GPU ops but pay-per-call and data leaves your network. Self-hosted = infra cost but full control. This is a security-and-cost architecture decision.
- **Fine-tuning potential** — sentence-transformers models can be fine-tuned on your own (query, tool) pairs from production. Hosted APIs can't.

**Full answer:** "Start with MiniLM for baseline, benchmark on a held-out set of (query, ground-truth-tool) pairs using recall@k. Consider moving to a larger open-source model or fine-tuning if recall@5 < ~90%. Weigh hosted APIs vs. self-hosted based on data sensitivity and cost projections."

</details>

<details>
<summary>Q2 — What to embed (click to expand)</summary>

Techniques a senior engineer should know:

- **HyDE (Hypothetical Document Embeddings) / query expansion** — generate what a good answer would look like, embed that instead.
- **Synthetic queries per tool** — use an LLM offline to generate 5-10 example user queries per tool. Embed those too. At search time, match against both the description and synthetic queries.
- **Hybrid search** — combine dense vector search with sparse keyword search (BM25). Dense catches semantics, sparse catches exact technical terms. Merge with reciprocal rank fusion (RRF).
- **Re-ranking** — after retrieving top-20 with embeddings, use a cross-encoder re-ranker (like `ms-marco-MiniLM-L-6-v2`) to re-score more carefully. Slower but much more accurate.
- **Metadata filtering** — if tools have tags/categories, filter the candidate pool first, then embed-search within that filtered set.

</details>

<details>
<summary>Q3 — Scaling (click to expand)</summary>

**When does linear scan break?** (384-dim float32):
- 10k vectors → ~15ms ✅
- 100k vectors → ~150ms, noticeable
- 1M+ vectors → seconds, unacceptable
- Memory: 1M × 384 × 4 bytes = ~1.5 GB just for embeddings

**What to swap in:** ANN (Approximate Nearest Neighbor) indexes, not a better linear scan:
- **HNSW** (Hierarchical Navigable Small World) — graph-based, ~95-99% recall, logarithmic search. Used by FAISS, Qdrant, Weaviate, pgvector.
- **IVF** (Inverted File Index) — clusters vectors, searches only nearest clusters. Less accurate than HNSW, lower memory.
- **Product Quantization (PQ)** — compresses vectors, usually combined with IVF or HNSW.

**Tradeoff:** give up exact nearest neighbors for ~100-1000x speedup. Fine for tool retrieval since the LLM does final selection anyway.

**Senior answer:** "At ~10k tools, move to FAISS with HNSW. At ~1M+, offload to a managed vector DB (Qdrant, pgvector) for persistence, replication, and filtering. The `ToolRegistry` interface lets me swap implementations without touching the orchestrator."

</details>

---

## Phase 2 — Orchestrator (~25 min)

Build the Orchestrator that does the following:

- Takes a user query as input
- Uses `ToolRegistry.search()` to get top-5 candidate tools
- Constructs a prompt with those tools' schemas and sends it to an LLM
- Parses the LLM's response to extract which tool(s) to call and with what arguments
- Calls those tools (you can mock the actual HTTP calls)
- Returns the final response

For the LLM, you can use the Anthropic API, OpenAI, or even a mock. Use **function calling / structured outputs** properly — don't rely on string parsing of free-form LLM output.

**Architectural constraint:** this orchestrator needs to eventually support **multi-turn tool calling** (LLM calls tool → gets result → decides to call another tool → etc.). Design your code now so you don't have to rewrite it when we add that.

### Phase 2 — Interview Questions

**Round 1 — A subtle bug and a design question:**

**1a.** You retrieve candidate tools once at the start of the turn using the original user query. What happens if the LLM calls a tool, gets a result, and then needs a different tool that wasn't in the top-5? Example: user says "help me ship this PR" → top-5 retrieves Git tools → LLM uses `git_diff` → realizes it needs to notify Slack → but `slack_notify` wasn't in the original top-5. What breaks, and how would you fix it?

**1b.** Your `_extract_text` has a bug. Look at the last return path — `messages[-1]["content"]` when iterations are exhausted. The last message in the loop is a user message with `tool_results` (a list of dicts), not assistant content blocks. What happens when you call `_extract_text` on that?

**Round 2 — Production concerns:**

**2a.** Your `ToolExecuter` calls `requests.post` synchronously. What are all the things wrong with this for a production orchestrator handling many concurrent agent sessions?

**2b.** You `print` errors in the executor. What should production code do instead?

**Round 3 — Architectural depth:**

**3a.** Right now everything is in one process. If you need to serve 10,000 concurrent agent sessions, how would you decompose this into separate services? Which pieces become their own service, why, and how do they communicate?

**3b.** Where does state live? If the orchestrator crashes mid-tool-call and the user reconnects — what happens? What would you add to make this resilient?

### Phase 2 — Key Takeaways

<details>
<summary>1a — Tool not in candidate set (click to expand)</summary>

Three options:
1. **Re-retrieve on miss** — when the LLM requests a tool not in candidates, do another vector search using the LLM's intent. Risky because it can spiral.
2. **Re-retrieve each iteration** — re-embed the latest assistant message and refresh candidates. Costs more but adapts to evolving intent.
3. **Two-tier retrieval** — use top-5 for the prompt, maintain a larger "available" set (top-20). Or expose a `search_tools(query)` **meta-tool** the LLM can call to discover more tools mid-conversation.

**Senior answer:** Option 3 — "tool of tools." Give the LLM agency to discover capabilities. This is how production agentic platforms work.

</details>

<details>
<summary>2a — Production concerns with synchronous requests.post (click to expand)</summary>

**Concurrency / I/O model:**
- Blocks the thread. Use `async/await` with `httpx.AsyncClient` or `aiohttp`.
- No connection pooling. Use a persistent client.

**Reliability:**
- No retry logic. Retry with exponential backoff + jitter on 5xx/timeouts only. Don't retry 4xx.
- No circuit breaker. Protect downstream from cascading failures.
- Single timeout for everything. Different tools have different SLAs — make it configurable per tool, and distinguish connect timeout from read timeout.
- No idempotency handling. Retrying a POST might create duplicates. Need idempotency keys for non-idempotent operations.

**Security:**
- No authentication. Real tools need OAuth, API keys, mTLS, per-tenant credentials.
- No request signing or audit trail.
- No input validation against the JSON schema before sending.

**Resource control:**
- No rate limiting per tool/tenant.
- No backpressure mechanism.
- Unbounded response size — a tool returning 50MB blows up the LLM context.

**Observability:**
- No structured logging or tracing. Each tool call should be a span with attributes.
- No metrics (latency p50/p95/p99, error rate, calls/sec per tool).

</details>

<details>
<summary>3a — Service decomposition (click to expand)</summary>

- **API Gateway / Session Service** — terminates connections, manages auth, holds session state.
- **Orchestrator Service** (stateless, horizontally scalable) — runs the agent loop.
- **Tool Registry Service** — owns tool metadata + embeddings. Read-heavy, cache aggressively.
- **Embedding Service** — wraps the model. Separate so you can scale on GPU nodes independently.
- **LLM Gateway** — proxy to LLM providers. Centralizes API keys, rate limiting, provider fallback, prompt caching.
- **Tool Executor Service** — invokes downstream tools. Queue-backed for backpressure. Handles retries, circuit breakers, credential injection.
- **State Store** — Redis (hot session state), Postgres (durable history).
- **Message Queue** — Kafka for async tool execution, audit logging, telemetry.
- **Observability stack** — OTel collector → traces (Tempo/Jaeger), metrics (Prometheus), logs (Clickhouse/Loki).

**Communication:** User-facing path is sync (HTTP/gRPC with timeouts). Everything else is async. Fire-and-forget for observability.

</details>

<details>
<summary>3b — State and resilience (click to expand)</summary>

**What state exists:** conversation history, tool call log, in-flight tool executions, iteration counter.

**Where it should live:** Conversation history → Redis (hot) + Postgres (durable, after each turn). In-flight tool calls → durable queue with session_id so another worker can pick up.

**Recovery semantics:** If the orchestrator crashes mid-tool-call, what guarantees do you give? At-most-once for non-idempotent tools (don't retry Jira ticket creation). At-least-once for idempotent ones. Enforce with idempotency keys and a tool-call ledger ("this tool_use_id was attempted at T, status=in_progress"). On recovery, check the ledger before retrying.

**Key phrase:** "durable state machine for the agent loop." Each step transition is persisted. Crash recovery resumes from the last persisted step.

</details>

---

## Phase 3 + 4 — Tracing & Safety Filter (~20 min)

Take your existing orchestrator and add two things:

1. **Basic OpenTelemetry-style tracing.** Create a simple `Tracer` class that records spans as nested dicts, with each span having: `name`, `start_time`, `end_time`, `attributes`, `status`, `children`. Wrap your orchestrator iterations, LLM calls, and tool calls in spans. The output should be a tree you could send to a real OTel collector later.

2. **An input safety filter.** Before sending the user query to the orchestrator, run it through a `SafetyFilter` that detects obvious prompt injection patterns. For now, simple regex-based detection is fine — look for things like "ignore previous instructions," "system:", etc. Return a structured result (`{passed: bool, reason: str}`) so a real ML-based filter could be swapped in.
