"""Distribution study: how do the vendor sets differ from the training data?

Compares train / val / vendor_a / vendor_b (gold labels only, no predictions) on
  1. doc-level shape (type, length, speaker tags, formatting, ASR noise)
  2. label-level counts (spans per doc, label share, span length)
  3. surface forms per label (date formats, dose formats, name formats, ...)
  4. vocabulary overlap with train (OOV rate per label)
  5. annotation conventions (random gold spans side by side, entity-looking
     text left unlabeled)

Run:  /Users/YSOWTIK/Documents/venvs/ml_torch/bin/python distribution_study.py
Writes the tables to results/distribution_tables.md. Seed: 0.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

SEED = 0
DATA = Path("data")
RESULTS = Path("results")
# README calls the vendor files vendor_a / vendor_b; on disk they are prospect_test_*.
FILES = {
    "train": DATA / "train.jsonl",
    "val": DATA / "val.jsonl",
    "vendor_a": DATA / "prospect_test_a.jsonl",
    "vendor_b": DATA / "prospect_test_b.jsonl",
}
LABELS = ["name", "drug_name", "drug_amount", "medical_condition", "date"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 40)

OUT_LINES = []  # everything printed as a table also goes to the markdown file


def show(title, df):
    """Print a table and remember it for the markdown file."""
    print(f"\n=== {title} ===")
    print(df.to_string())
    OUT_LINES.append(f"\n### {title}\n\n```\n{df.to_string()}\n```\n")


# ---------------------------------------------------------------- loading

def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def check_offsets(docs):
    """Guardrail: every gold span's offsets must reproduce its text."""
    bad = sum(d["text"][s["start"]:s["end"]] != s["text"] for d in docs for s in d["spans"])
    return bad


def doc_type(doc):
    """Transcript if it has 'ROLE:' speaker lines, note if it has **Header:** markdown.
    Judgment call: ids (trn-t-/trn-n-) agree for train/val; vendor_a ids (va-) carry
    no type, so the text markers decide."""
    text = doc["text"]
    if re.search(r"^[A-Z_]{4,}:", text, re.M):
        return "transcript"
    if "**" in text:
        return "note"
    return "other"


# ---------------------------------------------------------------- 1. doc level

SPEAKER_RE = re.compile(r"^([A-Z][A-Z_]{3,}):", re.M)
# Filler words typical of speech-to-text output.
FILLER_RE = re.compile(r"\b(um+|uh+|hmm+|er+m?|you know|i mean|like,)\b", re.I)


def doc_level(name, docs):
    rows = []
    for d in docs:
        t = d["text"]
        n_words = len(t.split())
        rows.append({
            "type": doc_type(d),
            "chars": len(t),
            "words": n_words,
            "markdown_bold": "**" in t,
            "bullets": bool(re.search(r"^\s*[-*] ", t, re.M)),
            "fillers_per_1k_words": 1000 * len(FILLER_RE.findall(t)) / max(n_words, 1),
            # share of alphabetic chars that are lowercase: ASR text is often all-lowercase
            "lower_share": sum(c.islower() for c in t) / max(sum(c.isalpha() for c in t), 1),
            # digits per 1k words: spoken numbers ("forty milligrams") have none
            "digits_per_1k_words": 1000 * sum(c.isdigit() for c in t) / max(n_words, 1),
        })
    df = pd.DataFrame(rows)
    return {
        "docs": len(docs),
        "transcripts": int((df["type"] == "transcript").sum()),
        "notes": int((df["type"] == "note").sum()),
        "median_chars": int(df["chars"].median()),
        "max_chars": int(df["chars"].max()),
        "median_words": int(df["words"].median()),
        "%docs_markdown": round(100 * df["markdown_bold"].mean(), 1),
        "%docs_bullets": round(100 * df["bullets"].mean(), 1),
        "fillers/1k_words": round(df["fillers_per_1k_words"].mean(), 1),
        "lowercase_share": round(df["lower_share"].mean(), 3),
        "digits/1k_words": round(df["digits_per_1k_words"].mean(), 1),
    }


def token_lengths(docs):
    """Word-piece count with the model's own tokenizer, to see how many docs need
    more than one 512-token window (510 content tokens + [CLS]/[SEP])."""
    tok = Tokenizer.from_file("model/tokenizer.json")
    n = pd.Series([len(tok.encode(d["text"]).ids) for d in docs])
    return {"median_tokens": int(n.median()), "max_tokens": int(n.max()),
            "%docs_>512_tokens": round(100 * (n > 512).mean(), 1)}


def speaker_tags(docs):
    return Counter(m for d in docs for m in SPEAKER_RE.findall(d["text"]))


# ---------------------------------------------------------------- 2. label level

def spans_frame(datasets):
    rows = []
    for name, docs in datasets.items():
        for d in docs:
            for s in d["spans"]:
                rows.append({"set": name, "doc": d["id"], "label": s["label"], "text": s["text"]})
    df = pd.DataFrame(rows)
    df["chars"] = df["text"].str.len()
    df["words"] = df["text"].str.split().str.len()
    return df


def label_level(spans, datasets):
    n_docs = pd.Series({k: len(v) for k, v in datasets.items()})
    counts = spans.pivot_table(index="label", columns="set", values="text", aggfunc="count", fill_value=0)
    counts = counts.reindex(index=LABELS, columns=list(datasets), fill_value=0)
    per_doc = (counts / n_docs).round(2)
    share = (100 * counts / counts.sum()).round(1)
    mean_words = spans.pivot_table(index="label", columns="set", values="words", aggfunc="mean")
    mean_words = mean_words.reindex(index=LABELS, columns=list(datasets)).round(2)
    return counts, per_doc, share, mean_words


# ---------------------------------------------------------------- 3. surface forms

NUMBER_WORDS = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|half)"
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"

# First matching rule wins, so order matters (most specific first).
DATE_FORMS = [
    ("MM/DD/YYYY", rf"^\d{{1,2}}/\d{{1,2}}/\d{{2,4}}$"),
    ("ISO YYYY-MM-DD", r"^\d{4}-\d{2}-\d{2}$"),
    ("Month D, YYYY", rf"^{MONTHS} \d{{1,2}}(st|nd|rd|th)?,? \d{{4}}$"),
    ("Month YYYY", rf"^{MONTHS},? \d{{4}}$"),
    ("Month D (no year)", rf"^{MONTHS} \d{{1,2}}(st|nd|rd|th)?$"),
    ("year only", r"^\d{4}$"),
    ("month only", rf"^{MONTHS}$"),
    ("relative", r"\b(ago|yesterday|today|tomorrow|tonight|last|next|past|this|in \w+ (day|week|month|year)s?|few|couple|since|week|month|morning|evening|night|weekend)\b"),
    ("weekday", r"^(on )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?$"),
]

AMOUNT_FORMS = [
    ("digit+unit, space (10 mg)", r"^\d[\d.,]*\s+[a-z/]+"),
    ("digit+unit, no space (10mg)", r"^\d[\d.,]*[a-z/]+"),
    ("spelled-out number", rf"\b{NUMBER_WORDS}\b"),
    ("bare number", r"^\d[\d.,]*$"),
]


def first_match(text, forms):
    t = text.lower().strip()
    for form, pattern in forms:
        if re.search(pattern, t):
            return form
    return "other"


def form_table(spans, label, forms):
    sub = spans[spans["label"] == label].copy()
    sub["form"] = sub["text"].map(lambda t: first_match(t, forms))
    tab = pd.crosstab(sub["form"], sub["set"], normalize="columns").mul(100).round(1)
    tab.loc["(n spans)"] = sub.groupby("set").size()
    return tab.reindex(columns=[c for c in FILES if c in tab.columns])


def span_flags(spans, label, flags):
    """For one label, % of spans in each set that match each regex flag."""
    sub = spans[spans["label"] == label]
    out = {}
    for flag, pattern in flags.items():
        hit = sub["text"].map(lambda t: bool(re.search(pattern, t)))
        out[flag] = hit.groupby(sub["set"]).mean().mul(100).round(1)
    out["(n spans)"] = sub.groupby("set").size()
    return pd.DataFrame(out).T.reindex(columns=[c for c in FILES if c in sub["set"].unique()])


NAME_FLAGS = {
    "title inside span (Dr./Mr./Ms.)": r"^(Dr|Mr|Mrs|Ms|Miss|Doctor)\b\.?",
    "single token": r"^\S+$",
    "all lowercase": r"^[^A-Z]+$",
    "ALL CAPS": r"^[A-Z][A-Z .'-]+$",
    "apostrophe or hyphen": r"['-]",
}
AMOUNT_FLAGS = {
    "unit mg / milligram": r"(?i)\b\d*\s*(mg|milligram)",
    "unit mcg/microgram": r"(?i)(mcg|microgram)",
    "unit units/UNT": r"(?i)(unit|UNT)",
    "unit ml": r"(?i)\bml\b|ML",
    "unit tablet/pill/puff/capsule": r"(?i)(tablet|pill|puff|capsule)",
    "frequency inside span": r"(?i)(daily|twice|once|every|bid|tid|qd|a day|per day|at night)",
}
DRUG_FLAGS = {
    "Capitalised first letter": r"^[A-Z]",
    "all lowercase": r"^[^A-Z]+$",
    "multi-word": r"\s",
    "hyphen combo (a-b)": r"-",
    "brackets [Brand]": r"\[",
}
CONDITION_FLAGS = {
    "abbreviation (2+ caps)": r"\b[A-Z]{2,}\b",
    "Capitalised first letter": r"^[A-Z]",
    "all lowercase": r"^[^A-Z]+$",
    "contains digit/stage": r"(?i)\d|stage",
    "spelled-out number": rf"(?i)\b{NUMBER_WORDS}\b",
}


# ---------------------------------------------------------------- 4. vocabulary overlap

def oov_table(spans):
    """% of each eval set's gold span strings (lowercased) never seen as a train
    gold span of the same label. Computed over span occurrences (tokens), and over
    distinct strings (types) as a second convention."""
    train_vocab = {
        label: set(spans[(spans["set"] == "train") & (spans["label"] == label)]["text"].str.lower())
        for label in LABELS
    }
    rows = {}
    for name in ["val", "vendor_a", "vendor_b"]:
        for label in LABELS:
            sub = spans[(spans["set"] == name) & (spans["label"] == label)]["text"].str.lower()
            if len(sub) == 0:
                rows[(label, name)] = {"n": 0, "OOV% occurrences": None, "OOV% distinct": None}
                continue
            unseen = ~sub.isin(train_vocab[label])
            distinct = set(sub)
            rows[(label, name)] = {
                "n": len(sub),
                "OOV% occurrences": round(100 * unseen.mean(), 1),
                "OOV% distinct": round(100 * len(distinct - train_vocab[label]) / len(distinct), 1),
            }
    df = pd.DataFrame(rows).T
    return df


# ---------------------------------------------------------------- 5. annotation conventions

def sample_spans(spans, label, k=6):
    rng = random.Random(SEED)
    cols = {}
    for name in FILES:
        pool = spans[(spans["set"] == name) & (spans["label"] == label)]["text"].tolist()
        picks = rng.sample(pool, min(k, len(pool)))
        cols[name] = picks + [""] * (k - len(picks))
    return pd.DataFrame(cols)


# Entity-looking text; we count matches NOT covered by any gold span.
LOOKALIKES = {
    "date: MM/DD/YYYY": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    "date: Month D, YYYY": rf"(?i)\b{MONTHS} \d{{1,2}},? \d{{4}}\b",
    "date: relative (yesterday/last week/ago/in N weeks)": r"(?i)\b(yesterday|(last|next) (week|month|year|night)|\w+ (days|weeks|months|years) ago|in \w+ (days|weeks|months|years))\b",
    "amount: digits + mg": r"(?i)\b\d[\d.,]*\s?mg\b",
    "amount: spelled + milligrams": rf"(?i)\b{NUMBER_WORDS}(?:[ -]{NUMBER_WORDS})* milligrams?\b",
    "name: Dr. Surname": r"\bDr\.? [A-Z][a-z]+",
}


def uncovered_lookalikes(docs, pattern):
    """Count regex matches, and how many are not overlapped by any gold span."""
    total = uncovered = 0
    examples = []
    for d in docs:
        for m in re.finditer(pattern, d["text"]):
            total += 1
            covered = any(s["start"] < m.end() and m.start() < s["end"] for s in d["spans"])
            if not covered:
                uncovered += 1
                if len(examples) < 2:
                    examples.append(f"{d['id']}: '{m.group(0)}'")
    return total, uncovered, examples


def near_duplicates(datasets):
    """Do any vendor docs share their exact text with train/val docs? (vendor_b ids
    use the same 'dev-n-' prefix as val, so check for leakage/overlap.)"""
    seen = {}
    for name in ["train", "val"]:
        for d in datasets[name]:
            seen[d["text"]] = f"{name}:{d['id']}"
    rows = {}
    for name in ["vendor_a", "vendor_b"]:
        hits = [d["id"] for d in datasets[name] if d["text"] in seen]
        ids_in_val = {d["id"] for d in datasets["val"]}
        rows[name] = {
            "exact text dup of train/val": len(hits),
            "id also used in val": sum(d["id"] in ids_in_val for d in datasets[name]),
        }
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    random.seed(SEED)
    datasets = {name: load(path) for name, path in FILES.items()}
    for name, docs in datasets.items():
        assert check_offsets(docs) == 0, f"{name}: span offsets do not round-trip"
    print("Loaded:", {k: len(v) for k, v in datasets.items()}, "- all span offsets round-trip")

    # 1. doc level
    show("Doc level", pd.DataFrame({n: doc_level(n, d) for n, d in datasets.items()}))
    tags = pd.DataFrame({n: speaker_tags(d) for n, d in datasets.items()}).fillna(0).astype(int)
    show("Speaker tags (count of 'ROLE:' line starts)", tags)
    train_tags = set(tags.index[tags["train"] > 0])
    for name in ["val", "vendor_a", "vendor_b"]:
        new = sorted(set(tags.index[tags[name] > 0]) - train_tags)
        print(f"  tags in {name} never seen in train: {new or 'none'}")
    show("Overlap / id collisions with train+val", near_duplicates(datasets))

    show("Model-tokenizer length", pd.DataFrame({n: token_lengths(d) for n, d in datasets.items()}))

    # 2. label level
    spans = spans_frame(datasets)
    counts, per_doc, share, mean_words = label_level(spans, datasets)
    show("Gold span counts", counts)
    show("Gold spans per doc", per_doc)
    show("Label share (% of spans)", share)
    show("Mean span length (words)", mean_words)

    # Fair comparison: vendor_a is all transcripts, vendor_b all notes, so compare
    # each with the train docs of the same type (judgment call: type from doc_type()).
    by_type = {
        "train_transcripts": [d for d in datasets["train"] if doc_type(d) == "transcript"],
        "vendor_a": datasets["vendor_a"],
        "train_notes": [d for d in datasets["train"] if doc_type(d) == "note"],
        "vendor_b": datasets["vendor_b"],
    }
    typed_spans = spans_frame(by_type)
    _, typed_per_doc, _, _ = label_level(typed_spans, by_type)
    show("Gold spans per doc, same doc type", typed_per_doc)
    typed_dates = typed_spans[typed_spans["label"] == "date"].copy()
    typed_dates["form"] = typed_dates["text"].map(lambda t: first_match(t, DATE_FORMS))
    show("date surface form, same doc type (% of date spans)",
         pd.crosstab(typed_dates["form"], typed_dates["set"], normalize="columns").mul(100).round(1))
    name_flags = {}
    for flag, pattern in NAME_FLAGS.items():
        names = typed_spans[typed_spans["label"] == "name"]
        hit = names["text"].map(lambda t: bool(re.search(pattern, t)))
        name_flags[flag] = hit.groupby(names["set"]).mean().mul(100).round(1)
    show("name flags, same doc type (% of name spans)", pd.DataFrame(name_flags).T)

    # 3. surface forms
    show("date: surface form (% of date spans)", form_table(spans, "date", DATE_FORMS))
    show("drug_amount: number format (% of spans)", form_table(spans, "drug_amount", AMOUNT_FORMS))
    show("drug_amount: flags (% of spans)", span_flags(spans, "drug_amount", AMOUNT_FLAGS))
    show("name: flags (% of spans)", span_flags(spans, "name", NAME_FLAGS))
    show("drug_name: flags (% of spans)", span_flags(spans, "drug_name", DRUG_FLAGS))
    show("medical_condition: flags (% of spans)", span_flags(spans, "medical_condition", CONDITION_FLAGS))

    # 4. vocabulary overlap
    show("OOV vs train gold spans of same label (lowercased, verbatim)", oov_table(spans))

    # Top unseen drug names in vendor_a: misspellings or genuinely new drugs?
    train_drugs = set(spans[(spans["set"] == "train") & (spans["label"] == "drug_name")]["text"].str.lower())
    for name in ["vendor_a", "vendor_b"]:
        va = spans[(spans["set"] == name) & (spans["label"] == "drug_name")]["text"].str.lower()
        unseen = va[~va.isin(train_drugs)].value_counts().head(12)
        print(f"\n  {name} drug_name strings not in train (top 12):", dict(unseen))

    # 5. annotation conventions
    for label in LABELS:
        show(f"Random gold spans, label={label} (seed {SEED})", sample_spans(spans, label))

    rows = {}
    for desc, pattern in LOOKALIKES.items():
        for name, docs in datasets.items():
            total, uncovered, ex = uncovered_lookalikes(docs, pattern)
            rows[(desc, name)] = {"matches": total, "not in any gold span": uncovered,
                                  "% unlabeled": round(100 * uncovered / total, 1) if total else None,
                                  "example": ex[0] if ex else ""}
    show("Entity-looking text left unlabeled", pd.DataFrame(rows).T)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "distribution_tables.md"
    out.write_text("# Distribution study - generated tables (distribution_study.py, seed 0)\n" + "".join(OUT_LINES))
    print(f"\nWrote {out}")
