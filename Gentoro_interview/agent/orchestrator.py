"""Home automation agent — main orchestrator loop.

Wires together: OpenAI LLM, MCP state server, RAG, conversation memory,
and OpenTelemetry tracing.
"""

import asyncio
import json
import os
import sys

# Ensure project root is on sys.path so package imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.memory import ConversationMemory
from agent.rag import RAG
from agent.telemetry import log_llm_io, tracer
from dotenv import load_dotenv

# MCP client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_MODEL = "gpt-4o-mini"

MCP_SERVER_CMD = sys.executable  # same Python interpreter
MCP_SERVER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mcp_server",
    "server.py",
)

SYSTEM_PROMPT = """\
You are a smart home automation assistant. You help the user monitor and
control their house through natural language.

RULES:
- You MUST use the provided tools to read or change any house state
  (temperature, doors, garage). Never guess state — always call a tool.
- Use the knowledge-base context provided below to ground your answers
  (user preferences, safety rules, house layout, device specs).
- Refer to previous conversation context when the user uses pronouns
  like "it" or references something mentioned earlier.
- Temperatures are in Celsius.  The user lives in Canada.
- If a request seems unusual or dangerous (e.g. very high temperature),
  warn the user before executing.

KNOWLEDGE-BASE CONTEXT:
{rag_context}
"""

# OpenAI tool definitions matching the MCP server tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Get the current and target temperature (°C) for a room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "description": "Room name, e.g. 'living_room' or 'bedroom'.",
                    }
                },
                "required": ["room"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_temperature",
            "description": "Set the target temperature (°C) for a room. Returns updated state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "description": "Room name, e.g. 'living_room' or 'bedroom'.",
                    },
                    "value": {
                        "type": "number",
                        "description": "Target temperature in Celsius.",
                    },
                },
                "required": ["room", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_door_status",
            "description": "Get the status ('open' or 'closed') of a door or garage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "door": {
                        "type": "string",
                        "description": "Door name, e.g. 'front_door', 'back_door', or 'garage'.",
                    }
                },
                "required": ["door"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_door_status",
            "description": "Open or close a door or garage. Returns new status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "door": {
                        "type": "string",
                        "description": "Door name, e.g. 'front_door', 'back_door', or 'garage'.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["open", "closed"],
                        "description": "'open' or 'closed'.",
                    },
                },
                "required": ["door", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_status",
            "description": "Get a full snapshot of all house devices (temperatures and doors).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_agent() -> None:
    """Main agent loop: readline → RAG → LLM → MCP tool calls → respond."""

    openai_client = OpenAI()

    # Initialize RAG
    print("Building RAG index over knowledge base …")
    rag = RAG(client=openai_client)
    rag.build_index()
    print(f"  Indexed {len(rag._chunks)} chunks.\n")

    # Initialize memory
    memory = ConversationMemory()

    # Connect to MCP state server via stdio
    server_params = StdioServerParameters(
        command=MCP_SERVER_CMD,
        args=[MCP_SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP state server.\n")
            print("Home Automation Agent ready. Type your message (or 'quit' to exit).\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                await handle_message(
                    user_input, session, openai_client, rag, memory
                )


async def handle_message(
    user_input: str,
    session: ClientSession,
    openai_client: OpenAI,
    rag: RAG,
    memory: ConversationMemory,
) -> None:
    """Process one user message end-to-end."""

    with tracer.start_as_current_span("agent.handle_message") as root_span:
        root_span.set_attribute("user.query", user_input[:512])

        # 1. Retrieve relevant KB context via RAG
        rag_results = rag.retrieve(user_input, top_k=3)
        rag_context = "\n---\n".join(
            f"[{r['source']} / {r['heading']}]\n{r['text']}" for r in rag_results
        )

        # 2. Add user message to memory
        memory.add_message("user", user_input)

        # 3. Build messages for OpenAI
        system_msg = SYSTEM_PROMPT.format(rag_context=rag_context)
        messages = [{"role": "system", "content": system_msg}]
        messages.extend(memory.get_context_window(max_messages=20))

        # 4. LLM loop (may involve multiple tool-call rounds)
        total_tokens = 0
        tools_called: list[str] = []

        for _round in range(10):  # safeguard against infinite loops
            with tracer.start_as_current_span("agent.llm_call") as llm_span:
                llm_span.set_attribute("llm.model", LLM_MODEL)
                llm_span.set_attribute("llm.round", _round)

                response = openai_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )

                choice = response.choices[0]
                usage = response.usage
                if usage:
                    total_tokens += usage.total_tokens
                    llm_span.set_attribute("llm.tokens.total", usage.total_tokens)

                # Log LLM I/O for hallucination auditing
                prompt_for_log = json.dumps(messages[-3:], default=str)[:4096]
                completion_for_log = (
                    choice.message.content or json.dumps(
                        [tc.function.name for tc in (choice.message.tool_calls or [])],
                        default=str,
                    )
                )
                log_llm_io(llm_span, prompt_for_log, completion_for_log)

            # If the model returns a text response, we're done
            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                assistant_text = choice.message.content or ""
                memory.add_message("assistant", assistant_text)
                root_span.set_attribute("agent.response", assistant_text[:512])
                root_span.set_attribute("agent.total_tokens", total_tokens)
                root_span.set_attribute("agent.tools_called", json.dumps(tools_called))
                print(f"\nAgent: {assistant_text}\n")
                return

            # Otherwise, execute tool calls via MCP
            messages.append(choice.message.model_dump())

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                tools_called.append(fn_name)

                with tracer.start_as_current_span("agent.mcp_tool_call") as tool_span:
                    tool_span.set_attribute("tool.name", fn_name)
                    tool_span.set_attribute("tool.args", json.dumps(fn_args))

                    try:
                        result = await session.call_tool(fn_name, fn_args)
                        # MCP result content is a list of content blocks
                        result_text = "\n".join(
                            getattr(block, "text", str(block))
                            for block in result.content
                        )
                        tool_span.set_attribute("tool.result", result_text[:1024])
                    except Exception as e:
                        result_text = f"Error: {e}"
                        tool_span.set_attribute("tool.error", str(e))
                        tool_span.set_status(
                            trace_api_status(e)
                        )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

        # If we exhaust rounds, respond with what we have
        print("\nAgent: I'm having trouble completing that request. Please try again.\n")


def trace_api_status(exc: Exception):
    """Create an OTel error status from an exception."""
    from opentelemetry.trace import Status, StatusCode
    return Status(StatusCode.ERROR, str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_agent())
