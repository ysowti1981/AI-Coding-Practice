# Distribution shift: train vs vendor A / vendor B

Source: `distribution_study.py` (seed 0), gold labels only, no predictions.
Full tables: `results/distribution_tables.md`. All span offsets round-trip in all four files.

Judgment calls:
- Doc type comes from text markers (`ROLE:` speaker lines = transcript, `**` markdown = note). Vendor A has
  100 transcripts and vendor B has 30 notes, so each vendor is compared with **train docs of the same type**
  wherever the doc-type mix would otherwise mislead (names per doc, dates).
- OOV = the lowercased gold span string never appears verbatim as a train gold span of the same label.
  Reported over occurrences (and over distinct strings in the tables).
- "Entity-looking text left unlabeled" uses regexes. They are heuristics that only find candidates.
- Vendor B ids reuse val's `dev-n-` prefix, but no id or text overlaps val or train (checked).

## Vendor A (100 transcripts)

1. **Dates are not annotated at all, but the text is full of relative dates.**
   e.g. va-001 "just started **yesterday**" has no gold span. Train labels this ("yesterday", "in 3 months").
   Magnitude: 0 date spans in vendor A vs 7.98 per doc in train transcripts (88.2% of them relative).
   356 of 357 relative-date regex hits in vendor A are unlabeled (99.7%). In train it is 7 of 799 (0.9%).
   This is a labeling-convention difference, not a model error: every date the model predicts on vendor A
   will count as a false positive.
2. **Doses are spelled out, not written as digits.**
   e.g. "ten milligrams", "three hundred milligrams", "forty milligrams". Train has "25 mg", "500 mg".
   Magnitude: 97.5% of vendor A drug_amount spans are spelled-out numbers vs 0.0% in train. OOV is 100.0%.
   Vendor A text has 0.0 digits per 1k words vs 68.9 in train.
3. **Drug names are misspelled (like ASR output).**
   e.g. "sertralene" (x14), "atorvastaten" (x13), "metoprol tartrate" (x12), "amlodipene", "metforman", "lysinopril".
   Magnitude: 20.4% of vendor A drug_name occurrences are unseen in train vs 0.9% for val (1.5% vendor B).
   Word-piece tokenisation will split these differently from the correct spelling.
4. **Conversation is noisier (disfluencies).**
   e.g. va-001 "like a little flutter, um, that's not normal"; "we've bin tracking".
   Magnitude: 13.9 filler words per 1k words vs 3.2 in train (4.9 in val).
   Condition spans with spelled-out numbers ("chronic kidney disease stage three"): 8.1% vs 0.0% in train.
5. **Not a shift (checked):** speaker tags are the same six roles as train. Names per doc (0.64 vs 0.72) and
   single-token names (100% vs 100%) match train transcripts. 43.8% of vendor A names are lowercase vs 0.0%,
   but the model is uncased, so that should not matter. All sets need more than one 512-token window
   (98.9% train, 100.0% vendor A). Vendor A is longer: median 1109 tokens vs 870.

## Vendor B (30 notes)

1. **Relative dates are not annotated. Only absolute dates are.**
   e.g. dev-n-001 "follow-up **in four weeks**", dev-n-032 "Diagnosed **2 years ago**": no gold span.
   Train notes label "in 6 months" style dates.
   Magnitude: 0.0% of vendor B date spans are relative vs 42.1% in train notes. 34 of 34 relative-date regex
   hits in vendor B are unlabeled (100.0%). In train it is 0.9%. Dates per doc: 2.67 vs 4.64 in train notes.
   Model date predictions on these phrases will count as false positives.
2. **Absolute date formats are shifted toward MM/DD/YYYY, plus birth years.**
   e.g. "10/12/2023" (DOB/encounter headers), "1985", "March 5, 1951".
   Magnitude: MM/DD/YYYY is 23.8% of date spans vs 8.0% in train notes. Year-only is 7.5% vs 0.8%.
   Date OOV is 36.2% vs 15.8% for val.
3. **More conditions per doc, and more abbreviations.**
   e.g. "ESRD", "Chronic Kidney Disease Stage 4", "Anemia of Chronic Kidney Disease".
   Magnitude: 9.17 condition spans per doc vs 7.06 in train notes. 24.4% are abbreviations vs 9.6% in all
   of train. But condition OOV is only 2.9%, so the vocabulary is familiar.
4. **Dose units lean to injectables (mL, UNT/ML).**
   e.g. "4000 UNT/ML", "1 ML", "0.0272 MG/MG".
   Magnitude: 52.3% of drug_amount spans contain ml vs 15.4% in train. 36.0% contain units/UNT vs 12.4%.
   Drug_amount OOV is only 2.3%, so this is mild.
5. **Otherwise close to train notes:** drug_name OOV is 1.5%, name spans per doc are 5.63 vs 5.75,
   and "Dr. Surname" is always labeled (70 of 70). Vendor B is a small sample (30 docs, 80 date spans), so
   the per-label percentages are noisy.

## Summary

- **Vendor A** has real input shift (spelled-out doses, misspelled drugs, disfluent speech) and a date
  convention that differs from ours (nothing labeled).
- **Vendor B** looks like our notes. Its main difference is the convention: relative dates are not labeled.
- Before blaming the model for date errors on either vendor, ask the vendors whether relative dates are in scope.
