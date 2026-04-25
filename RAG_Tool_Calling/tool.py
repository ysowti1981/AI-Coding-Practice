from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Tool:
    """Represents  a registered tool with its name, description, and parameters."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    endpoint_url: str = ""

    def embedding_text(self) -> str:
        """Generates a text representation of the tool for embedding purposes."""
        return f"{self.name}: {self.description}"
