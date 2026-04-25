from RAG_Tool_Calling.tool import Tool
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
        name="send_email",
        description="Send an email to a recipient with a subject and body.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
        endpoint_url="https://api.example.com/email",
    ))

    registry.register(Tool(
        name="search_flights",
        description="Search for available flights between two airports on a given date.",
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string", "format": "date"},
            },
            "required": ["origin", "destination", "date"],
        },
        endpoint_url="https://api.example.com/flights",
    ))

    results = registry.search("What's the temperature in Paris today?", top_k=2)
    for t in results:
        print(t.name, "->", t.description)


if __name__ == "__main__":
    main()
