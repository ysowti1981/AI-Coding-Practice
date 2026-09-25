#!/usr/bin/env python3
"""Standalone NER prediction with an int8 ONNX model. Dependencies: onnxruntime, numpy, tokenizers (no torch).

Usage: python predict.py --model model --input data/vendor_a.jsonl --output preds/vendor_a.jsonl
       python predict.py --model model --text "Dr. Alvarez started the patient on aspirin 325 mg on October 18, 2023 for hypertension."
Import: from predict import predict_string; spans = predict_string("...")   # loads ./model next to this file once
Output JSONL: {"id", "text", "spans": [{"label","start","end","text"}]} with character offsets on the ORIGINAL text.
Long documents are split into overlapping 512-token windows; per-token logits are averaged across windows.
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def load_model(model_dir: str, quantized=True):
    d = Path(model_dir)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    tok.no_padding(); tok.no_truncation()
    id2label = {int(k): v for k, v in json.load(open(d / "id2label.json")).items()}
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(d / ("model_int8.onnx" if quantized else "model_fp32.onnx")), so, providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    if cls is None: cls, sep = tok.token_to_id("<s>"), tok.token_to_id("</s>")
    return tok, sess, id2label, input_names, cls, sep


def predict_doc(text, tok, sess, id2label, input_names, cls, sep, max_len=512, stride=128):
    enc = tok.encode(text, add_special_tokens=False)
    ids, offs = enc.ids, enc.offsets
    body, step = max_len - 2, max_len - 2 - stride
    acc = {}  # (start,end) char offset -> summed logits
    for st in range(0, max(1, len(ids) - stride), step):
        en = min(st + body, len(ids))
        w = [cls] + ids[st:en] + [sep]
        feed = {"input_ids": np.array([w], dtype=np.int64), "attention_mask": np.ones((1, len(w)), dtype=np.int64)}
        if "token_type_ids" in input_names: feed["token_type_ids"] = np.zeros((1, len(w)), dtype=np.int64)
        logits = sess.run(None, feed)[0][0]
        for ti, (a, b) in enumerate(offs[st:en], start=1):
            if b > a:
                acc[(a, b)] = acc[(a, b)] + logits[ti] if (a, b) in acc else logits[ti].copy()
        if en == len(ids): break
    spans, cur = [], None
    for (a, b), v in sorted(acc.items()):
        lab = id2label[int(v.argmax())]
        if lab == "O":
            if cur: spans.append(cur); cur = None
            continue
        bio, ent = lab.split("-", 1)
        if bio == "B" or cur is None or cur["label"] != ent:
            if cur: spans.append(cur)
            cur = {"label": ent, "start": a, "end": b}
        else:
            cur["end"] = b
    if cur: spans.append(cur)
    out = []
    for s in spans:
        while s["end"] > s["start"] and text[s["end"] - 1].isspace(): s["end"] -= 1
        while s["end"] > s["start"] and text[s["start"]].isspace(): s["start"] += 1
        if s["end"] > s["start"]:
            s["text"] = text[s["start"]:s["end"]]; out.append(s)
    return out


# ---- importable API --------------------------------------------------------------------------------------
_MODEL_CACHE = {}
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"   # the packet ships model/ next to this file


def load_model_cached(model_dir=None, quantized=True):
    """Load (once) and cache the tokenizer + ONNX session for `model_dir` (default: ./model next to this script)."""
    key = (str(model_dir or DEFAULT_MODEL_DIR), quantized)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(key[0], quantized=quantized)
    return _MODEL_CACHE[key]


def predict_string(text, model_dir=None, quantized=True, max_len=512, stride=128):
    """Run NER on one string and return its spans: [{"label", "start", "end", "text"}, ...] with character offsets on `text`.

    >>> from predict import predict_string
    >>> predict_string("Call Dr. Alvarez about the aspirin 325 mg.")
    """
    tok, sess, id2label, input_names, cls, sep = load_model_cached(model_dir, quantized)
    return predict_doc(text, tok, sess, id2label, input_names, cls, sep, max_len=max_len, stride=stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--input"); ap.add_argument("--output"); ap.add_argument("--text")
    ap.add_argument("--fp32", action="store_true", help="use model_fp32.onnx instead of the int8 model")
    args = ap.parse_args()
    m = load_model_cached(args.model, quantized=not args.fp32)
    if args.text:
        print(json.dumps(predict_doc(args.text, *m), indent=1)); return
    docs = [json.loads(l) for l in open(args.input) if l.strip()]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    with open(args.output, "w") as f:
        for d in docs:
            spans = predict_doc(d["text"], *m)
            counts.update(s["label"] for s in spans)
            f.write(json.dumps({"id": d["id"], "text": d["text"], "spans": spans}, ensure_ascii=False) + "\n")
    print(f"{len(docs)} docs -> {args.output}; predicted spans by label: {dict(counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
