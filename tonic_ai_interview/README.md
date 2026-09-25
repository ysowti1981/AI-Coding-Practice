# NER model review — vendor data

## The situation

We ship a small NER model that finds five kinds of entities in clinical text: patient/clinician **names**, **drug names**, **drug amounts** (dose or strength with its unit), **medical conditions**, and **dates**. It was trained on our own clinical notes and conversation transcripts and it does well on our validation data.

Two vendors have now sent us labeled samples of their transcripts and notes and asked how our model performs on their data before we process the rest. Your job over the next hour: evaluate the model on the vendor data, figure out whether there are performance problems and what is causing them, and propose — and where you can, demonstrate — how you would address them.

## What is in this folder

```
data/
  train.jsonl      372 docs   the data the model was trained on (186 notes + 186 transcripts)
  val.jsonl         40 docs   our held-out validation set (20 notes + 20 transcripts)
  vendor_a.jsonl   100 docs   vendor A's labeled sample (transcripts)
  vendor_b.jsonl    30 docs   vendor B's labeled sample (notes)
preds/
  val.jsonl, vendor_a.jsonl, vendor_b.jsonl    the model's predictions on those three files, same format
model/             the model: int8 ONNX graph, tokenizer, label map, config
predict.py         run the model:  python predict.py --model model --input data/vendor_a.jsonl --output out.jsonl
                   or from Python: from predict import predict_string; predict_string("some text")
example_predict.py a short example of predict_string
```

Every line in every JSONL file is one document:

```json
{"id": "va-001", "text": "...", "spans": [{"label": "drug_name", "start": 27, "end": 34, "text": "aspirin"}, ...]}
```

`start`/`end` are character offsets into `text` (end exclusive), so `text[start:end] == span["text"]`. The `preds/` files use exactly the same format with the model's spans in place of the gold ones — you can regenerate them with `predict.py` at any time (it takes a few seconds).

## About the model

`model/` is a fine-tuned `google/bert_uncased_L-4_H-256_A-4` (BERT-Mini, 11M parameters) token classifier, trained on `data/train.jsonl` with the five labels above, using 512-token windows for long documents and early stopping on `data/val.jsonl`. No data augmentation was used. It was exported to ONNX and dynamically quantized to int8 for CPU inference. `predict.py` handles windowing and returns character spans on the original text.

Dependencies for `predict.py`: `onnxruntime`, `numpy`, `tokenizers` — no PyTorch. The quickest way to run it without touching your environment:

```bash
uv run --python 3.12 --with onnxruntime,numpy,tokenizers python predict.py --model model --text "aspirin 325 mg daily"
```

## What we would like you to do

1. Decide how the model should be evaluated on this data, and evaluate it on both vendor sets (and on `val.jsonl` for comparison).
2. Walk us through the evaluation: what the numbers are, how you computed them, and how you would interpret them for the vendors.
3. Identify which labels, if any, are problematic on each vendor's data.
4. Diagnose why.
5. Propose how you would address each problem — and if a fix can be tried within the hour, try it and measure it.

## How the session works

- **60 minutes.** Rough guide: about half on evaluation and diagnosis, about half on fixes and discussion. We would rather you go deep on what you find than cover everything.
- **Use whatever tools you normally use**, including AI coding agents. We are interested in how you decide what to do and how you check it, so please think out loud and tell us what you are choosing *not* to do and why.
- **Ask questions.** Your interviewer is standing in for the vendors and for the team that trained the model, and will answer questions about the data, the labeling, and the model's history.
- **Retraining the model is out of scope** for the hour (no GPU here). Describing what you would retrain with is very much in scope.
- Nothing in this folder is real patient data — all of it is synthetic.
