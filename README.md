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
