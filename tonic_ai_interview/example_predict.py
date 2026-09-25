#!/usr/bin/env python3
"""Example: using predict_string() from predict.py.

Run from the ner-debug/ folder (so `predict.py` and `model/` are importable):

    python example_predict.py

Dependencies: onnxruntime, numpy, tokenizers  (e.g. `uv run --with onnxruntime,numpy,tokenizers example_predict.py`)
"""
import json

from predict import predict_string

# 1) A single string -> list of spans. Offsets are character positions in the string you passed.
text = "Dr. Alvarez started the patient on aspirin 325 mg on October 18, 2023 for hypertension."
print(f"text: {text}")
spans = predict_string(text)
print("spans:")
for s in spans:
    print(json.dumps(s, indent=2))

