import os
from typing import Any, Dict, List

import anthropic

from RAG_Tool_Calling.tool import Tool


class LLMClient:
    """A simple wrapper around the Anthropic API to interact with Claude."""

    def __init__(self, model: str = "claude-opus-4-5", max_tokens: int = 1024):

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _format_tools(tools: List[Tool]) -> List[Dict[str, Any]]:
        """Formats the list of tools into a structure suitable for the LLM."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters
            }
            for tool in tools
        ]

    def complete(
        self,
        messages: List[Dict[str, str]],
        tools: List[Tool],
        system: str = ""
    ) -> Any:
        """Sends a completion request to the LLM with the given messages and tools."""

        tool_definitions = self._format_tools(tools)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "You are a helpful assistant that can call tools to answer user queries and has access to tools.",
            tools=tool_definitions,
            messages=messages
        )

        return response