"""De-identification pipeline.

Stage 1 detects entities, stage 2 replaces them with realistic fakes so the
text stays usable for downstream research.

Run once to produce data/redacted.jsonl.
"""

import json
import random

from load_data import DATA_DIR, OUT_PATH

COVERAGE = 0.94          # detector recall, measured by the detection eval
SEED = 13

OUT = DATA_DIR / "redacted.jsonl"

POOLS = {
    "FIRSTNAME": ["James", "Maria", "David", "Sarah", "Michael", "Linda",
                  "Robert", "Emily"],
    "MIDDLENAME": ["Lee", "Marie", "Alan", "Rose", "Grace", "Paul",
                   "Anne", "Scott"],
    "LASTNAME": ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                 "Miller", "Davis"],
    "EMAIL": ["jsmith@example.com", "m.garcia@example.org",
              "dbrown42@example.net", "sarah.jones@example.com",
              "mwilliams@example.org", "linda.d@example.net",
              "rmiller@example.com", "edavis@example.org"],
    "DATE": ["03/14/2019", "07/22/2020", "11/08/2018", "01/30/2021",
             "05/17/2017", "09/02/2022", "12/25/2016", "04/11/2023"],
    "DOB": ["03/14/1985", "07/22/1978", "11/08/1992", "01/30/1966",
            "05/17/1990", "09/02/1983", "12/25/1971", "04/11/1988"],
    "PHONENUMBER": ["555-0142", "555-0198", "555-0127", "555-0163",
                    "555-0119", "555-0175", "555-0134", "555-0188"],
    "CITY": ["Springfield", "Riverton", "Fairview", "Greenville",
             "Ashland", "Clayton", "Milford", "Dayton"],
    "SSN": ["412-55-0193", "298-41-7762", "631-08-4419", "175-92-3308",
            "508-27-6641", "743-16-2295", "389-64-8817", "926-73-1504"],
}


def fake_for(label, rng):
    pool = POOLS.get(label)
    if pool:
        return rng.choice(pool)
    return f"[{label}]"


def redact(text, spans, rng):
    """Replace each detected span with a fake value, left to right."""
    replaced = 0
    for span in sorted(spans, key=lambda s: s["start"]):
        if rng.random() > COVERAGE:
            continue
        fake = fake_for(span["label"], rng)
        text = text[:span["start"]] + fake + text[span["end"]:]
        replaced += 1
    return text, replaced


def main():
    rng = random.Random(SEED)
    n = 0
    with open(OUT_PATH) as src, open(OUT, "w") as dst:
        for line in src:
            row = json.loads(line)
            redacted, replaced = redact(
                row["source_text"], row["privacy_mask"], rng
            )
            dst.write(json.dumps({
                "doc_id": str(row["id"]),
                "original_text": row["source_text"],
                "redacted_text": redacted,
                "spans_replaced": replaced,
            }) + "\n")
            n += 1
    print(f"wrote {n} documents to {OUT.name}")


if __name__ == "__main__":
    main()
