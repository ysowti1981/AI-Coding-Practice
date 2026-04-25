from typing import Any, Dict, List

from RAG_Tool_Calling.llm_client import LLMClient
from RAG_Tool_Calling.tool_executer import ToolExecuter
from RAG_Tool_Calling.tool_registry import ToolRegistry


class Orchestrator:
    """Drives the full agent loop:
    user query -> search tools -> LLM decides which tool to call -> execute tool -> feed result back to LLM -> repeat until done.

    Designed so that multi-turn conversations are just "keep calling run()
    with an accumulated message history" — the internal tool-use loop
    already handles multi-step reasoning within one user turn.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        llm: LLMClient,
        executer: ToolExecuter,
        top_k: int = 5,
        max_iterations: int = 10,
    ):

        self.tool_registry = registry
        self.llm_client = llm
        self.tool_executer = executer
        self.top_k = top_k
        self.max_iterations = max_iterations

    def run(
        self,
        query: str,
        history: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """
        Process one user turn.

        Args:
            query: the user's natural-language input for this turn.
            history: prior messages from earlier turns, if any. Passing this
                     in is what enables multi-turn conversations — the caller
                     owns the history and feeds it back on each call.

        Returns:
            A dict with:
              - 'response': the assistant's final text
              - 'messages': the updated message history (caller should save
                            this and pass it as `history` next turn)
              - 'tool_calls': list of (tool_name, arguments, result) for
                              debugging/logging
        """

        messages: List[Dict[str, Any]] = list(history) if history else []
        messages.append({"role": "user", "content": query})

        # Retrieve relevant tools ONCE per user turn.
        # Rationale: semantic relevance is driven by user intent, which is
        # stable within a turn. Re-retrieving per iteration would add latency
        # without meaningfully improving tool selection.

        candidate_tools = self.tool_registry.search(query, top_k=self.top_k)
        # build a quick lookup so we can resolve names -> tool objects
        tool_by_name = {tool.name: tool for tool in candidate_tools}

        tool_call_log: List[Dict[str, Any]] = []

        # Inner loop: the LLM may need multiple tool calls to answer one query.
        # This is distinct from multi-turn (which is multiple user messages).

        for _ in range(self.max_iterations):
            response = self.llm_client.complete(messages, candidate_tools)

            messages.append({"role": "assistant", "content": response.content})
            # stop_reson = "end_turn" means the model is done - no more tools
            if response.stop_reason == "end_turn":
                final_text = self._extract_text(response.content)
                return {
                    "response": final_text,
                    "messages": messages,
                    "tool_calls": tool_call_log
                }

            # Otherwise, execute every tool_use block the model emitted.
            # The model can request multiple tools in parallel in a single turn.

            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool = tool_by_name.get(block.name)
                if tool is None:
                    # Model hallucinated a tool not in our candidate set.
                    # Report the error back so it can recover.
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Tool '{block.name}' not found in registry.",
                        "is_error": True,
                    })
                    continue
                else:
                    result = self.tool_executer.execute(tool, block.input)
                    result_str = str(result) if not isinstance(result, str) else result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

                    tool_call_log.append({
                        "tool_name": block.name,
                        "arguments": block.input,
                        "result": result
                    })

            # Feed tool results back as a user message. This is the
            # Anthropic-specified format for returning tool outputs.
            messages.append({"role": "user", "content": tool_results})

        # If we hit the max iteration limit, return what we have with a warning.
        final_text = self._extract_text(messages[-1]["content"])
        return {
            "response": final_text,
            "messages": messages,
            "tool_calls": tool_call_log,
            "warning": f"Max iterations ({self.max_iterations}) reached without 'end_turn'."
        }

    @staticmethod
    def _extract_text(content: List[Any]) -> str:
        """Pull plain text out of Anthropic's content blocks."""
        return "".join(blocks.text for blocks in content if blocks.type == "text")
