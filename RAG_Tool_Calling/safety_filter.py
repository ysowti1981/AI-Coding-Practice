import re
from typing import Dict

# Patterns that indicate prompt injection attempts.
_INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?previous\s+instructions", "prompt override attempt"),
    (r"ignore\s+(all\s+)?above\s+instructions", "prompt override attempt"),
    (r"disregard\s+(all\s+)?previous", "prompt override attempt"),
    (r"forget\s+(all\s+)?(your\s+)?instructions", "prompt override attempt"),
    (r"you\s+are\s+now\s+", "role reassignment attempt"),
    (r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions", "restriction bypass attempt"),
    (r"pretend\s+you\s+are\s+", "role reassignment attempt"),
    (r"^\s*system\s*:", "system prompt injection"),
    (r"<\s*system\s*>", "system prompt injection"),
    (r"\]\s*system\s*:", "system prompt injection"),
    (r"do\s+not\s+follow\s+(your\s+)?guidelines", "guideline bypass attempt"),
    (r"override\s+(your\s+)?(safety|content)\s+(filter|policy)", "safety bypass attempt"),
    (r"jailbreak", "jailbreak attempt"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), reason) for p, reason in _INJECTION_PATTERNS]


class SafetyFilter:
    """Regex-based input safety filter for detecting prompt injection.

    Returns a structured result so it can be swapped for an ML-based
    filter later without changing the interface.
    """

    def check(self, text: str) -> Dict[str, object]:
        """Screen user input for injection patterns.

        Returns:
            {"passed": True}  if the input looks safe.
            {"passed": False, "reason": "..."} if a pattern matched.
        """
        for pattern, reason in _COMPILED:
            if pattern.search(text):
                return {"passed": False, "reason": reason}
        return {"passed": True, "reason": ""}
