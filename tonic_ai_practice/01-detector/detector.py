import re

PATTERNS = {
    "EMAIL":      re.compile(r"[\w.]+@[\w.]+"),
    "PHONE":      re.compile(r"\d{3}-\d{3}-\d{4}"),
    "SSN":        re.compile(r"\d{3}-\d{2}-\d{4}"),
    "DATE":       re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}"),
    "CREDITCARD": re.compile(r"\d{4} \d{4} \d{4} \d{4}"),
    "ZIP":        re.compile(r"\b\d{5}\b"),
}

PERSON_RE = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")

def detect(text):
    """Return a list of {'start', 'end', 'label'} character spans."""
    spans = []
    for label, pat in PATTERNS.items():
        for m in pat.finditer(text):
            spans.append({"start": m.start(), "end": m.end(), "label": label})
    for m in PERSON_RE.finditer(text):[]
        spans.append({"start": m.start(), "end": m.end(), "label": "PERSON"})
    return spans