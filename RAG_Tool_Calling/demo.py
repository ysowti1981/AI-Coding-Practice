import os

from dotenv import load_dotenv

# Load .env into os.environ BEFORE importing anything that needs the key.
# Done at the entry point only — library modules stay side-effect-free.
load_dotenv()

# Fail fast with a clear message if the key is missing, rather than
# surfacing a cryptic auth error mid-request.
if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. "
        "Create a .env file with ANTHROPIC_API_KEY=your-key"
    )

from RAG_Tool_Calling.llm_client import LLMClient
from RAG_Tool_Calling.orchestrator import Orchestrator
from RAG_Tool_Calling.tool import Tool
from RAG_Tool_Calling.tool_executer import ToolExecuter
from RAG_Tool_Calling.tool_registry import ToolRegistry


def main():
    registry = ToolRegistry()
    registry.register(Tool(
        name="get_weather",
        description="Get the current weather and temperature for a given city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        endpoint_url="https://api.example.com/weather",
    ))
    registry.register(Tool(
        name="search_flights",
        description="Search for available flights between two airports on a given date.",
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": ["origin", "destination", "date"],
        },
        endpoint_url="https://api.example.com/flights",
    ))

    orchestrator = Orchestrator(
        registry=registry,
        llm=LLMClient(),
        executer=ToolExecuter(),
    )

    # Turn 1
    result = orchestrator.run("What's the weather in Paris?")
    print("Assistant:", result["response"])

    # Turn 2 — pass back the accumulated history for multi-turn.
    result = orchestrator.run(
        "And find me a flight there from SFO tomorrow.",
        history=result["messages"],
    )
    print("Assistant:", result["response"])


if __name__ == "__main__":
    main()
