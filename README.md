# AI Coding Practice

## Problem 1: Tool Orchestration Service

Build a service that sits between an LLM-based agent and a registry of hundreds of enterprise tools — things like "create a Jira ticket," "query Salesforce contacts," "send a Slack message," etc. Each tool has a name, a natural language description, and a JSON schema for its parameters.

When the agent receives a user request like _"find all open deals over $50k and notify the sales team on Slack,"_ the service needs to:

1. **Identify which tools are relevant** (semantic matching)
2. **Plan the execution order** (e.g., Salesforce query first, then Slack notification)
3. **Execute with proper error handling and observability**

### Architecture

The solution is organized into these components:

| Component | File | Responsibility |
|---|---|---|
| **Tool** | `tool.py` | Dataclass representing a single tool (name, description, parameters, endpoint URL) |
| **Tool Registry** | `tool_registry.py` | Stores tools and performs semantic search using sentence-transformers embeddings to find relevant tools for a query |
| **LLM Client** | `llm_client.py` | Wrapper around the Anthropic API for chat completions with tool-use support |
| **Tool Executer** | `tool_executer.py` | Executes tool calls by POSTing to tool endpoints |
| **Orchestrator** | `orchestrator.py` | Drives the full agent loop: user query → semantic tool search → LLM decides tool calls → execute → feed results back → repeat until done |
| **Tracer** | `tracer.py` | OpenTelemetry-style tracing — records nested spans (name, timing, attributes, status, children) for orchestrator iterations, LLM calls, and tool calls |
| **Safety Filter** | `safety_filter.py` | Regex-based input filter that detects prompt injection patterns (e.g. "ignore previous instructions"). Returns `{passed, reason}` so an ML-based filter can be swapped in |
| **Demo** | `demo.py` | Entry point demonstrating multi-turn conversation with tool use, tracing, and safety filtering |

### How It Works

1. Tools are registered in the `ToolRegistry`, which embeds their descriptions using a sentence-transformer model.
2. On each user turn, the `Orchestrator` semantically searches for the top-k relevant tools.
3. The `LLMClient` sends the user message and candidate tools to Claude, which decides whether to call tools or respond directly.
4. If tools are called, the `ToolExecuter` hits the tool endpoints and results are fed back to the LLM.
5. The loop continues until the LLM produces a final text response.
6. A `SafetyFilter` screens each user query before it reaches the orchestrator — flagged inputs are blocked with a structured reason.
7. A `Tracer` records the full span tree (orchestrator run → iterations → LLM calls / tool calls) for observability.

### Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API key:

```
ANTHROPIC_API_KEY=your-key-here
```

### Run

```bash
python RAG_Tool_Calling/demo.py
```

### Claude Shared Link
https://claude.ai/share/5e8688e6-001c-4d50-949b-f879387c15c1

---

## Problem 2: Home Automation AI Agent

Build an AI agent that lets users control and monitor their house through natural language. The agent must integrate MCP (Model Context Protocol) for state management, RAG for domain knowledge, persistent conversational memory, and full OpenTelemetry observability.

Given prompts like _"What is the current temperature in the living room?"_ or _"I just left, did I forget anything open?"_, the agent needs to:

1. **Communicate with a state server** exclusively through MCP tool calls (never access state directly)
2. **Retrieve domain knowledge** (house layout, device specs, safety rules, user preferences) via RAG over markdown files
3. **Maintain conversation memory** so users can reference earlier context (e.g., "set *it* to 22")
4. **Enforce safety rules** (e.g., reject temperatures outside 15–30 °C)
5. **Trace the full lifecycle** with OpenTelemetry spans for LLM calls, MCP tool invocations, RAG retrieval, and memory operations

### Architecture

The solution is organized into these components:

| Component | File | Responsibility |
|---|---|---|
| **Orchestrator** | `agent/orchestrator.py` | Main agent loop — wires LLM, MCP, RAG, memory, and tracing |
| **Memory** | `agent/memory.py` | Sliding-window conversational memory for multi-turn context |
| **RAG** | `agent/rag.py` | Chunks, embeds, and retrieves from `kb/*.md` using OpenAI embeddings and cosine similarity |
| **Telemetry** | `agent/telemetry.py` | OpenTelemetry tracer setup and LLM I/O logging for hallucination auditing |
| **MCP State Server** | `mcp_server/server.py` | FastMCP server exposing device tools (`get_temperature`, `set_temperature`, `get_door_status`, `set_door_status`, `get_all_status`) |
| **State Store** | `mcp_server/state.py` | In-memory house state (temperatures, doors, garage) — single source of truth |
| **Knowledge Base** | `kb/*.md` | Markdown files describing user profile, house layout, device specs, and safety rules |

### How It Works

1. The RAG pipeline indexes all `kb/*.md` files by splitting on `##` headings and embedding each chunk with OpenAI's `text-embedding-3-small`.
2. The MCP state server launches as a subprocess over stdio, exposing tools for reading/writing house device state.
3. On each user turn, the orchestrator retrieves the top-k relevant knowledge chunks via RAG and injects them into the LLM system prompt.
4. The LLM (GPT-4o-mini) decides whether to call MCP tools or respond directly. Tool results are fed back for further reasoning.
5. Conversational memory retains the last N messages so the agent can resolve references like "set *it* to 22" or "the living room is freezing."
6. OpenTelemetry traces every operation — LLM calls, MCP tool invocations, RAG retrieval, and memory access — for debugging and hallucination auditing.

### Setup

```bash
cd Gentoro_interview
pip install -r requirements.txt
```

Create a `.env` file with your API key:

```
OPENAI_API_KEY=sk-your-key-here
```

### Run

```bash
cd Gentoro_interview
python -m agent.orchestrator
```
