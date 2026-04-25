# MCP Tool Generation Pipeline

## Problem

Build an offline pipeline that ingests an OpenAPI specification and produces a registered set of validated, agent-ready MCP tools. A new customer pastes in an OpenAPI spec for their internal HR system — say it has 80 endpoints. Within minutes, those 80 endpoints should become clean MCP tools that AI agents can call reliably.

**Why this is hard:**

- OpenAPI specs in the wild are messy — inconsistent naming, missing descriptions, weird parameter shapes, vendor extensions, `$ref` chains.
- LLMs work better with agent-friendly tool descriptions ("Search for employees by department") than raw API endpoint names ("GET /v2/emp/idx").
- You can't just generate and ship — bad tools cause agents to fail in production. There need to be quality gates.
- This needs to scale to many customers, many specs, many tools, and the customer expects this to be fast and reliable.

---

## Phase 0 — System Design (Verbal, 0–15 min)

Walk through the high-level architecture of this pipeline:

1. What are the major stages from spec ingestion to registered tool?
2. What components/services exist?
3. Where does the LLM fit in, and where does it *not* fit in?
4. What are the quality gates?

### Phase 0 — Interview Questions

**Q1 — Stage decomposition:**
What are the concrete stages? Where is each stage deterministic vs. generative? Why does that distinction matter for reliability?

**Q2 — LLM boundaries:**
Which parts of this pipeline should use an LLM, and which should be pure code? What happens if you let the LLM do too much? What happens if you let it do too little?

**Q3 — Quality gates:**
What does "validated" mean here? What kinds of failures can a generated tool have, and how do you catch them before production?

**Q4 — Latency & cost:**
80 endpoints × LLM calls. How do you keep this under 2 minutes? What's the cost profile? How do you handle a spec with 500 endpoints?

---

## Phase 1 — Spec Parser & Schema Resolver (~20 min)

Build the deterministic front-end of the pipeline:

- **`SpecParser`** — takes a raw OpenAPI spec (JSON or YAML), resolves all `$ref` references, and extracts a flat list of endpoint descriptors. Each descriptor should contain: `method`, `path`, `operationId` (if present), `summary`, `description`, `parameters`, `requestBody` schema, `response` schema.
- **`SchemaResolver`** — recursively resolves `$ref` chains, handles circular references gracefully, and flattens nested schemas into self-contained JSON schemas suitable for MCP tool parameter definitions.

**No LLM usage in this phase.** This is pure parsing and transformation.

### Phase 1 — Interview Questions

**Q1 — $ref resolution:**
How do you handle circular `$ref` references? (e.g., `Employee` → `manager: $ref Employee`). What does your output look like?

**Q2 — Schema diversity:**
OpenAPI supports `oneOf`, `anyOf`, `allOf`, `discriminator`, `additionalProperties`, `patternProperties`. How do you map these to MCP tool parameter schemas that an LLM can understand and fill correctly?

**Q3 — Vendor extensions:**
Some specs have `x-internal: true` or `x-deprecated` on endpoints. How do you decide what to include vs. filter out?

**Q4 — Validation at this stage:**
What can go wrong with the input spec itself? How do you give the customer useful error messages vs. just crashing?

---

## Phase 2 — LLM-Powered Tool Generator (~20 min)

Build the generative core of the pipeline:

- **`ToolGenerator`** — takes the list of endpoint descriptors from Phase 1 and uses an LLM to produce MCP tool definitions. Each tool definition has: `name` (clean, agent-friendly), `description` (natural language, tells the agent when to use this tool), `parameters` (JSON schema), `endpoint` mapping back to the original API.
- Use **structured outputs** (JSON mode or function calling) — don't parse free-form text.
- Batch endpoints intelligently — don't make 80 individual LLM calls if you can group related endpoints.

### Phase 2 — Interview Questions

**Q1 — Prompt design:**
Show me your prompt. What context do you give the LLM? How do you prevent it from hallucinating parameters that don't exist in the spec?

**Q2 — Structured outputs:**
How do you enforce that the LLM output conforms to your expected schema? What happens when it doesn't? How many retries before you flag for human review?

**Q3 — Batching strategy:**
How do you decide which endpoints to batch together? What's the tradeoff between batch size and output quality? What about token limits?

**Q4 — Hallucination risks:**
The LLM might: invent parameters, merge two endpoints, give a description that doesn't match the actual behavior, generate invalid JSON schema. How do you catch each of these?

**Q5 — Determinism:**
Two runs on the same spec should produce the same tools. How do you handle LLM non-determinism? Does it matter?

---

## Phase 3 — Evaluation Harness (~25 min)

Build an offline evaluation system that tests generated tools against a golden set:

- **`GoldenDataset`** — a set of (endpoint_descriptor, expected_tool_definition) pairs, manually curated. Start with 10-15 examples.
- **`ToolEvaluator`** — compares generated tools against golden tools and scores them on multiple dimensions:
  - **Name quality** — is the name clear, consistent, agent-friendly?
  - **Description quality** — does it accurately describe what the tool does and when to use it?
  - **Schema correctness** — do the parameters match the original API? No hallucinated fields? No missing required fields?
  - **Schema completeness** — are all parameters captured with proper types, descriptions, and constraints?
- Support both **automated metrics** (schema diff, field coverage) and **LLM-as-judge** scoring for subjective quality (name/description quality).
- Output a structured report: per-tool scores, aggregate metrics, regressions from previous runs.

### Phase 3 — Interview Questions

**Q1 — LLM-as-judge:**
You're using an LLM to judge LLM output. What are the failure modes? How do you validate the judge itself? What if the judge disagrees with your golden set?

**Q2 — Metrics design:**
What's your primary metric? If you had to pick one number to decide "ship or don't ship this batch of tools," what is it and why?

**Q3 — Regression testing:**
A new prompt version improves 70 tools but degrades 3. How do you detect and handle this? What's your CI/CD gate?

**Q4 — Feedback loops:**
In production, agents will call these tools. Some calls will fail, some will succeed. How do you turn production traces into eval data? How do you close the loop from "tool failed in production" → "tool gets fixed"?

**Q5 — Golden dataset maintenance:**
Who creates the golden set? How do you keep it representative as new specs come in? How many examples do you need?

---

## Phase 4 — Distributed Execution & Multi-Tenancy (Discussion + Light Coding, ~20 min)

Take the offline pipeline and discuss how to make it run as a multi-tenant service:

- **`JobQueue`** — accepts spec-processing jobs, assigns them to workers, tracks status.
- **`Worker`** — pulls jobs, runs the pipeline, reports results. Must be idempotent and isolated between tenants.
- Discuss: job schema, worker lifecycle, failure handling, tenant isolation, priority queues, rate limiting per tenant.

### Phase 4 — Interview Questions

**Q1 — Idempotency:**
Customer submits the same spec twice. Or the worker crashes mid-pipeline and the job gets re-queued. What happens? How do you ensure idempotent behavior across the full pipeline?

**Q2 — Tenant isolation:**
Two customers submit specs simultaneously. How do you ensure they can't see each other's tools, and one customer's huge spec doesn't starve another customer's small spec?

**Q3 — Cost accounting:**
Each pipeline run costs LLM tokens. How do you track, limit, and bill per-tenant usage? What happens when a customer hits their limit mid-pipeline?

**Q4 — Failure modes:**
The LLM provider rate-limits you. A spec has 500 endpoints and the pipeline takes 10 minutes. The worker OOMs on a massive nested schema. Walk me through each failure and your mitigation.

**Q5 — SLA design:**
Customer expects tools within 2 minutes. You have 50 customers submitting specs at once. How do you design the SLA? What levers do you have (more workers, smaller batches, priority tiers)?

---

## Phase 5 — Wrap-up (5 min)

- Questions for the interviewer
- Debrief on architecture decisions and tradeoffs
