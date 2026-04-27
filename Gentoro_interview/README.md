# Home Automation AI Agent

An intelligent home automation agent that lets users control and monitor their house through natural language. Built with MCP, RAG, persistent memory, stateful device management, and full OpenTelemetry observability.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Key Techniques](#key-techniques)
- [Setup & Installation](#setup--installation)
- [Running the Agent](#running-the-agent)
- [Example Interactions](#example-interactions)
- [Configuration](#configuration)
- [Dependencies](#dependencies)

---

## Problem Statement

Build an AI Agent that integrates the following techniques:

| Technique | Requirement |
|---|---|
| **MCP or A2A** | Agent ↔ Server communication via a structured protocol |
| **Memory** | Retain conversational context so the user can refer back to earlier parts of the thread |
| **RAG** | Ground the agent in domain-specific knowledge using a retrieval-augmented generation pipeline |
| **State** | Maintain live state of house devices — accessed **only** through MCP tools, never directly by the orchestrator |
| **OpenTelemetry** | Instrument the full request lifecycle to help developers catch errors, hallucinations, and performance issues |
| **API** | Use an external LLM API for reasoning and response generation |

### Scenario

A **home automation AI Agent** that:

1. Accepts natural language questions and commands from the user.
2. Communicates with the state server through **MCP** (Model Context Protocol).
3. Maintains **memory** of the conversation so users can reference items mentioned earlier in the thread.
4. Uses a **knowledge base** (RAG over markdown files) to understand house context — layout, device specs, user preferences, and safety rules.
5. Keeps **state** on house elements (temperature, doors, garage) and exposes it exclusively through MCP tools.
6. Is fully instrumented with **OpenTelemetry** to provide traces, spans, and events across the entire pipeline.

### Example Prompts

| Prompt | Expected Behavior |
|---|---|
| *"What is the current temperature in the living room?"* | Agent calls `get_temperature` via MCP and returns the current reading. |
| *"Set temperature to 36"* | RAG retrieves user profile (lives in Canada, uses Celsius) and safety rules (max 30 °C). Agent warns this is dangerously high. |
| *"OMG, the living room is freezing"* | Using **memory** (living room was just discussed) and **state** (current temp), the agent increases the temperature. |
| *"I have just left, did I forget anything open?"* | RAG provides house layout (doors, garage). **State** provides their open/closed status. Agent reports what's still open. |

### Constraints

- State **must** be accessed through MCP tools — the orchestrator never reads or writes state directly.
- Manage at least two categories of house devices: **temperature** and **doors**.
- OpenTelemetry must provide signals that help developers catch **errors**, **hallucinations**, and **operational problems**.
- RAG raw material can be simple **markdown files**.

---

## Architecture Overview

```
                         ┌──────────────────┐
                         │   OpenAI API     │
                         │  (gpt-4o-mini)   │
                         └────────▲─────────┘
                                  │ Chat Completions
                                  │ + Embeddings
┌─────────────┐        ┌──────────┴───────────────────────────────────────┐
│             │        │              Orchestrator (Agent)                │
│  User CLI   │◄──────►│                                                  │
│             │        │  ┌────────────┐  ┌─────────┐  ┌──────────────┐  │
│  (stdin/    │        │  │   Memory   │  │   RAG   │  │ OpenTelemetry│  │
│   stdout)   │        │  │            │  │         │  │   (Tracing)  │  │
│             │        │  │ Multi-turn │  │ kb/*.md │  │              │  │
└─────────────┘        │  │ context    │  │ chunks  │  │ Spans, Events│  │
                       │  └────────────┘  └─────────┘  └──────────────┘  │
                       └──────────────┬──────────────────────────────────-┘
                                      │
                                      │ MCP Tool Calls (stdio transport)
                                      │
                              ┌───────▼───────┐
                              │  State Server  │
                              │  (FastMCP)     │
                              │                │
                              │ ┌────────────┐ │
                              │ │ Temperature│ │
                              │ │ • living   │ │
                              │ │ • bedroom  │ │
                              │ ├────────────┤ │
                              │ │   Doors    │ │
                              │ │ • front    │ │
                              │ │ • back     │ │
                              │ │ • garage   │ │
                              │ └────────────┘ │
                              └────────────────┘
```

### Components

| Component | Role |
|---|---|
| **Orchestrator** | LLM-powered agent that interprets user requests, plans actions, and coordinates all subsystems |
| **MCP Protocol** | Communication layer between the orchestrator and the state server |
| **State Server** | MCP server that owns all house device state (temperature, doors, garage) — the *only* way to read or mutate state |
| **Memory** | Conversational memory that retains context across the interaction thread |
| **RAG** | Retrieval-Augmented Generation over a knowledge base of markdown files describing the house, user preferences, and device specs |
| **OpenTelemetry** | Traces, metrics, and logs across the entire request lifecycle for debugging errors, hallucinations, and performance issues |
| **LLM API** | External LLM API (OpenAI `gpt-4o-mini`) used by the orchestrator for reasoning and response generation |

---

## Project Structure

```
Gentoro_interview/
├── requirements.txt            # Python dependencies
├── agent/                      # Core agent logic
│   ├── __init__.py
│   ├── orchestrator.py         # Main agent loop — wires LLM, MCP, RAG, memory, and tracing
│   ├── memory.py               # ConversationMemory class for multi-turn context
│   ├── rag.py                  # RAG pipeline — chunks, embeds, and retrieves from kb/
│   └── telemetry.py            # OpenTelemetry tracer setup and LLM I/O logging helper
├── mcp_server/                 # MCP state server (runs as a subprocess over stdio)
│   ├── __init__.py
│   ├── server.py               # FastMCP server exposing device tools
│   └── state.py                # In-memory house state store (single source of truth)
└── kb/                         # Knowledge base (markdown files indexed by RAG)
    ├── device_specs.md          # Thermostat, smart lock, and garage door specs
    ├── house_layout.md          # Room, door, and garage layout with device locations
    ├── safety_rules.md          # Temperature limits, door/garage safety rules
    └── user_profile.md          # User identity, location, and preferences
```

---

## Key Techniques

### 1. MCP (Model Context Protocol)

The agent communicates with the **State Server** exclusively through MCP tool calls. The state server runs as a **subprocess** connected via **stdio** transport. State is never accessed directly by the orchestrator — all reads and writes go through well-defined MCP tools exposed by the state server.

The orchestrator uses `mcp.client.stdio.stdio_client` to launch and connect to the server (`mcp_server/server.py`), and calls tools via `session.call_tool(fn_name, fn_args)`.

**Exposed MCP Tools:**

| Tool | Description |
|---|---|
| `get_temperature(room)` | Returns the current and target temperature (°C) for a given room |
| `set_temperature(room, value)` | Sets the target temperature for a room; returns updated state |
| `get_door_status(door)` | Returns `"open"` or `"closed"` status of a door or garage |
| `set_door_status(door, status)` | Opens or closes a door or garage; returns new status |
| `get_all_status()` | Returns full snapshot of all house devices |

Valid rooms: `living_room`, `bedroom`
Valid doors: `front_door`, `back_door`, `garage`

### 2. Memory

Conversation history is managed by the `ConversationMemory` class (`agent/memory.py`). It stores messages in a list and provides a sliding context window (default last 20 messages) to fit the LLM context window. This enables:

- Referring back to previously mentioned rooms or devices ("set *it* to 22")
- Contextual reasoning ("the living room is freezing" → agent recalls it was just asked about the living room temperature)
- Coherent multi-turn interactions without repeating context

### 3. RAG (Retrieval-Augmented Generation)

The `RAG` class (`agent/rag.py`) implements a simple in-memory RAG pipeline:

1. **Indexing:** Reads all `kb/*.md` files, splits them into chunks by `##` headings, and embeds each chunk using OpenAI's `text-embedding-3-small` model.
2. **Retrieval:** For each user query, embeds the query and computes cosine similarity against all chunk embeddings, returning the top-k (default 3) most relevant chunks.
3. **Augmentation:** The retrieved chunks are injected into the LLM system prompt under `KNOWLEDGE-BASE CONTEXT`, grounding the agent's responses in house-specific facts.

**Knowledge base files:**

| File | Contents |
|---|---|
| `kb/user_profile.md` | User identity (Alex Martin, Ottawa, Canada); household members; preferred units (metric); schedule |
| `kb/house_layout.md` | Rooms, doors, garage layout; device locations; leaving-the-house checklist |
| `kb/device_specs.md` | SmartTemp Pro 3000 thermostat (15–30 °C range), SecureLock Z-200 smart locks, GarageMaster 500 opener |
| `kb/safety_rules.md` | Temperature limits (15–30 °C), unusual temp warnings (>26 °C), door/garage safety rules, night-time checks |

### 4. State Management

All house device state lives in the **State Server** (`mcp_server/state.py`) as an in-memory Python dictionary. It is accessed only through MCP tools.

**Initial state:**

```json
{
  "temperature": {
    "living_room": { "current": 21.5, "target": 22.0 },
    "bedroom": { "current": 20.0, "target": 20.0 }
  },
  "doors": {
    "front_door": "closed",
    "back_door": "closed",
    "garage": "open"
  }
}
```

The state module validates all inputs — unknown room/door names raise `ValueError`, and door status must be `"open"` or `"closed"`.

> **Important:** The orchestrator must never hold or cache state directly. Every state query or mutation goes through the MCP state server.

### 5. OpenTelemetry Observability

The `telemetry.py` module sets up a `TracerProvider` with a `ConsoleSpanExporter` under the service name `home-agent`. Every significant operation is traced:

- `agent.handle_message` — root span for each user interaction
- `agent.llm_call` — each LLM API call (model, round, token usage)
- `agent.mcp_tool_call` — each MCP tool invocation (name, args, result)
- `rag.build_index`, `rag.embed_batch`, `rag.retrieve` — RAG pipeline spans
- `memory.add_message`, `memory.get_context_window` — memory operations
- `mcp.get_temperature`, `mcp.set_temperature`, etc. — server-side tool spans

LLM prompts and completions are also logged as span events via `log_llm_io()` for hallucination auditing.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Install

```bash
cd Gentoro_interview
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the `Gentoro_interview/` directory (or export the variable):

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

The orchestrator loads this via `python-dotenv`.

---

## Running the Agent

```bash
cd Gentoro_interview
python -m agent.orchestrator
```

This will:
1. Build the RAG index over the `kb/` markdown files (embedding all chunks).
2. Launch the MCP state server as a subprocess over stdio.
3. Start an interactive CLI loop where you can type natural language commands.

Type `quit`, `exit`, or `q` to stop.

---

## Example Interactions

```
You: What's the temperature in the living room?
Agent: The living room is currently at 21.5°C with a target of 22.0°C.

You: Set it to 24
Agent: Done — the living room target temperature is now set to 24.0°C.

You: Is the garage open?
Agent: Yes, the garage is currently open. Would you like me to close it?

You: Close it please
Agent: The garage is now closed.

You: Set bedroom to 35
Agent: ⚠️ Warning: 35°C exceeds the maximum safe indoor temperature of 30°C.
       The SmartTemp Pro 3000 supports a range of 15–30°C. Would you like to
       set it to a lower temperature instead?

You: I'm leaving the house
Agent: Let me check everything for you...
       - Front door: closed ✓
       - Back door: closed ✓
       - Garage: closed ✓
       - I've set both thermostats to energy-saving mode (18°C).
       Have a good day!
```

---

## Configuration

| Setting | Location | Default |
|---|---|---|
| LLM model | `agent/orchestrator.py` → `LLM_MODEL` | `gpt-4o-mini` |
| Embedding model | `agent/rag.py` → `EMBED_MODEL` | `text-embedding-3-small` |
| RAG top-k results | `agent/orchestrator.py` → `rag.retrieve(top_k=)` | `3` |
| Context window size | `agent/orchestrator.py` → `memory.get_context_window(max_messages=)` | `20` messages |
| Max tool-call rounds | `agent/orchestrator.py` → loop limit | `10` rounds |
| OTel service name | `agent/telemetry.py` → resource | `home-agent` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `mcp[cli]>=1.0.0` | MCP client/server framework (FastMCP, stdio transport) |
| `openai>=1.0.0` | OpenAI API client for chat completions and embeddings |
| `opentelemetry-api>=1.20.0` | OpenTelemetry tracing API |
| `opentelemetry-sdk>=1.20.0` | OpenTelemetry SDK (TracerProvider, ConsoleSpanExporter) |
| `numpy>=1.24.0` | Vector operations for cosine similarity in RAG |
| `python-dotenv>=1.0.0` | Load `.env` file for API keys |

---

## Example Interactions

### "What is the current temperature now in the living room?"

1. Orchestrator receives the user message and appends it to **memory**.
2. Calls MCP tool `get_temperature("living_room")` on the state server.
3. State server returns `{ "current": 21.5, "target": 22.0 }`.
4. Orchestrator responds: *"The living room is currently 21.5 °C, with the target set to 22 °C."*

### "Set temperature to 36"

1. Orchestrator appends to memory; queries **RAG** for user context.
2. RAG returns `user_profile.md` → user lives in Canada, uses Celsius.
3. Orchestrator reasons that 36 °C is extremely high for a home and warns the user.
4. If confirmed, calls MCP tool `set_temperature("living_room", 36)`.
5. Responds: *"⚠️ 36 °C is very high for indoor heating. Are you sure? I've set it, but please confirm."*

### "OMG, the living room is freezing"

1. Orchestrator checks **memory** — previous context mentions the living room.
2. Calls MCP tool `get_temperature("living_room")` → current is 15 °C, target is 16 °C.
3. Orchestrator decides to increase the temperature and calls `set_temperature("living_room", 22)`.
4. Responds: *"The living room was at 15 °C — I've raised the target to 22 °C. It should warm up shortly."*

### "I have just left, did I forget anything open?"

1. Orchestrator queries **RAG** for what "open" things matter → `house_layout.md` mentions front door, back door, and garage.
2. Calls MCP tool `get_all_status()` on the state server.
3. State returns garage is **open**, all doors are closed.
4. Responds: *"Your garage is still open! All doors are closed. Would you like me to close the garage?"*

---

## Project Structure

```
.
├── README.md
├── agent/
│   ├── orchestrator.py          # Main agent loop: LLM reasoning, tool dispatch
│   ├── memory.py                # Conversation memory management
│   ├── rag.py                   # RAG pipeline: indexing & retrieval over KB
│   └── telemetry.py             # OpenTelemetry setup: tracer, meter, logger
├── mcp_server/
│   ├── server.py                # MCP state server exposing house device tools
│   └── state.py                 # In-memory house state store
├── kb/                          # Knowledge base (RAG source material)
│   ├── user_profile.md
│   ├── house_layout.md
│   ├── device_specs.md
│   └── safety_rules.md
├── requirements.txt
└── .env.example                 # API keys and config
```

---

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key (or compatible LLM provider)

### Installation

```bash
git clone <repo-url>
cd interview_gentoro
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API key and OpenTelemetry exporter endpoint
```

### Running

```bash
# Start the MCP state server
python mcp_server/server.py

# In another terminal, start the agent
python agent/orchestrator.py
```

---

## Observability

Export traces and metrics to any OpenTelemetry-compatible backend (Jaeger, Grafana Tempo, etc.):

```bash
# Example: export to a local Jaeger instance
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 python agent/orchestrator.py
```

Key things to monitor:

- **Trace waterfall** — see the full flow from user prompt to final response, including LLM and MCP calls.
- **LLM prompt/completion logs** — audit for hallucinations by comparing RAG-retrieved context against the model's output.
- **MCP tool errors** — failed state reads/writes surface immediately as error spans.
- **Latency metrics** — identify slow RAG retrievals or LLM calls.
