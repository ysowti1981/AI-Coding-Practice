# CLAUDE.md — live interview

## Context

A one-hour live technical interview. Screen-shared, on my laptop, AI tools
allowed. The problem is data science and evaluation: a dataset, something
provided to evaluate (a model, a pipeline, predictions, a synthetic dataset),
and a decision someone needs made.

**I narrate while you work.** The interviewer reads your output as it appears,
and may read this file.

## The one rule

If I can't defend a line when the interviewer asks "why does this do that?",
the line is a liability. Optimise for my understanding and my ability to say
it out loud, not for code polish.

## Environment

- **Use the `ml_torch` venv for everything, by full path, every time:**

  ```bash
  /Users/YSOWTIK/Documents/venvs/ml_torch/bin/python script.py
  ```

  **Do not rely on `source …/activate`.** Each shell command runs in a fresh
  shell, so activation does not carry over to the next command, and plain
  `python` resolves to a *different* venv (`ml`, pandas 2.3) that gives
  different results. If you do activate, do it in the same command:

  ```bash
  source /Users/YSOWTIK/Documents/venvs/ml_torch/bin/activate && python script.py
  ```

  For me in a terminal, `source /Users/YSOWTIK/Documents/venvs/ml_torch/bin/activate`
  once is fine — the terminal keeps it.
- **First two minutes:** run the version check below with the full path, and
  confirm every provided file loads. Nothing else until both succeed. The
  executable printed must be `…/venvs/ml_torch/bin/python`.

  ```bash
  /Users/YSOWTIK/Documents/venvs/ml_torch/bin/python -c "import sys, pandas, numpy, sklearn, scipy; print(sys.executable); print(pandas.__version__, numpy.__version__, sklearn.__version__, scipy.__version__)"
  ```
- Installed: pandas 3, numpy, scikit-learn, scipy, `datasets`, `certifi`.
  **Ask before installing anything.** If the problem implies a library
  constraint, confirm it with the interviewer rather than assuming.
- **pandas 3:** string columns use a new string dtype, and `astype(str)` can
  behave differently around missing values. Build row keys with `.map(str)`,
  and treat `NaN` as an explicit value where it matters.
- If a download fails SSL verification (macOS python.org builds), set
  `SSL_CERT_FILE` to `certifi.where()`. **Never disable verification.**

## How to work with me

- **Explain before you write.** Two or three sentences on the approach and the
  trade-off, then the code. Those sentences are what I say out loud.
- **Small increments.** One function or cell, run it, stop.
- **Say what a step is for, then give its result.** One line on what it tests
  and which question it answers, before any number.
- **Put the deciding number first.** The figure that moves the decision, in
  plain units ("four in five high-risk entities get through"), then the
  supporting table.
- **One claim per point, in this shape:**

  ```
  <claim, one sentence>
    e.g.  <one real case from the data>
    say:  "<the sentence I speak>"
  ```

- **Introduce a new concept before using it**: what it measures, how it's
  computed, a tiny example on this data, and why this one and not the obvious
  alternative. Three to five lines live; not an essay.
- **Flag judgment calls** as you make them: label mapping, matching rules,
  what to drop, which denominator, which columns. These are what interviewers
  ask about.
- **The verb is the scope.** "Explain" means explain, not run. "Look at the
  data" doesn't mean build the evaluator. If the interviewer interrupts or asks
  something, **stop and answer that**. Don't finish the current step first.
- **Push back** if my approach won't finish in the time left.
- **Keep screen output short.** Summaries, and at most about 20 rows. No full
  dumps. Warn me before anything that runs longer than about 30 seconds.

## Time

Sixty minutes, wall clock:

| Phase | Budget | Deliverable |
|---|---:|---|
| Look at the data | 10 min | Schema, joins verified, 5–10 real examples read, baseline or control identified |
| Reproduce the claim | 10 min | The headline number everyone is relying on, verified or refuted |
| Dig | 20 min | A specific claim about a specific failure, measured against a baseline |
| Decide | 10 min | The answer, in the reader's terms |
| Fix or next step | 10 min | One fix measured, or the one number to re-measure |

**Cut scope, not the clock.** You can't see the time. I'll call out time
checks; when I do, tell me which phase we're in and what to cut.

## One notes file

`NOTES.md`, created at the first measurement and updated as we go. Not a
report, a running record:

1. **Top:** the question in the reader's own words. The plan (steps and what
   to cut first). **What I'd need to see to believe the headline claim**,
   written before any number exists.
2. **Middle:** each finding as it's produced: what it tested, the number, and
   what it means.
3. **End:** the answer, written for whoever asked.

Mark it **incomplete** until it isn't. Anything important said in chat goes in
the file; a finding that only exists in scrollback is lost.

## Method: questions to put to any headline metric

- **Which stage does it measure?** The thing actually being shipped, or an
  earlier stage of the pipeline?
- **Which convention?** Matching rule, micro vs macro, which columns,
  label-strict or not. Two defensible conventions can rank things in opposite
  orders.
- **What's the baseline?** What does chance look like? What does a second real
  sample look like?
- **Is the difference bigger than noise?** Resample the independent unit;
  pair the comparison when both sides were scored on the same items. Report
  intervals. When an interval contains the null, say *"indistinguishable
  from X"*.
- **What do the errors cost?** Are misses and false alarms symmetric? Weight
  by severity where outcomes differ.
- **What unit does the reader care about?** Spans, documents, people, rows?
- **What can this metric not see, by construction?**

## Guardrails

- Confirm joins are total, and ids and offsets round-trip, **before** computing
  any metric.
- Never modify the source data in place.
- Every number I state must come from code that ran in this session. Re-run
  after any edit.
- Don't repeat a number from the brief as our own until we've reproduced it.
- Reconcile totals: bucket counts sum to item counts, and tables agree with
  each other.
- If a preprocessing step changes a denominator (merging, filtering), state the
  before and after.
- **One `if __name__ == "__main__":` block, at the bottom of the file.** Add
  new functions above it, never append below it. Or work in a notebook.
- Fix seeds and state them.
- When a result is surprising, look at three raw examples before explaining it.

## The deliverable

The last ten minutes produce the answer, not more analysis:

- **The decision**, in the reader's terms, in one line.
- **Two or three pieces of evidence** that force it.
- **What would change it**, or the one number to re-measure.
- **The fix to do first**, and whether it changes the decision.

If there are several readers (say, a technical lead and a non-technical
stakeholder), write one reply per reader, in the form each one asked for.

## Keep this file clean

Process and working style only. No problem-specific answers and no pre-written
solutions.
