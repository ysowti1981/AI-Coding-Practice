from typing import Any, Dict

import requests

from RAG_Tool_Calling.tool import Tool


class ToolExecuter:
    """Responsible for actually involking a tool's endpoint.

    Separated from the orchestrator so that:
      - We can mock it in tests.
      - We can later add retries, timeouts, auth, caching, etc., in one place.
      - The orchestrator stays focused on the LLM loop.
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def execute(self, tool: Tool, arguments: Dict[str, Any]) -> Any:
        """
        POST the arguments to the tool's endpoint and return the parsed result.
        We return a string/JSON payload that can be fed back to the LLM.
        """

        try:
            response = requests.post(tool.endpoint_url, json=arguments, timeout=self.timeout)
            response.raise_for_status()
            # return the json response if possible, otherwise return the raw text
            try:
                return response.json()  # Assuming the tool returns JSON. Adjust if needed.
            except ValueError:
                return response.text

        except requests.RequestException as e:
            # Handle exceptions (e.g., log them, return an error message, etc.)
            print(f"Error executing tool '{tool.name}': {e}")
            return f"Error executing tool '{tool.name}': {e}"
