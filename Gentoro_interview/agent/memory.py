"""Conversation memory for the home automation agent."""

from __future__ import annotations

from agent.telemetry import tracer


class ConversationMemory:
    """Stores the message history for a single conversation thread."""

    def __init__(self) -> None:
        self._messages: list[dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        with tracer.start_as_current_span("memory.add_message") as span:
            span.set_attribute("memory.role", role)
            span.set_attribute("memory.content_length", len(content))
            self._messages.append({"role": role, "content": content})
            span.set_attribute("memory.total_messages", len(self._messages))

    def get_messages(self) -> list[dict[str, str]]:
        """Return the full conversation history."""
        return list(self._messages)

    def get_context_window(self, max_messages: int = 20) -> list[dict[str, str]]:
        """Return the last *max_messages* messages to fit the LLM context window."""
        with tracer.start_as_current_span("memory.get_context_window") as span:
            window = self._messages[-max_messages:]
            span.set_attribute("memory.window_size", len(window))
            span.set_attribute("memory.total_messages", len(self._messages))
            return list(window)
