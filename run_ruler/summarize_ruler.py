#!/usr/bin/env python3
"""Summarize RULER results written by run/test_ruler.py.

Each ``<task>.jsonl`` row carries a ShadowKV-style per-task ``score`` (computed
in run/test_ruler.py). This script reports per-task means and the macro-average
over the full 13-task ShadowKV RULER set, flagging any missing tasks so the mean
stays comparable to a 13-task baseline.

Usage:
    python run_ruler/summarize_ruler.py <out_dir> [<out_dir2> ...]
"""
import json
import os
import sys

FULL_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]


def _legacy_score(row):
    """Fallback for rows without a precomputed score (substring match)."""
    pred = str(row.get("pred", ""))
    answers = row.get("answer") or row.get("answers") or row.get("outputs") or []
    if not isinstance(answers, list):
        answers = [answers]
    return 1.0 if any(str(a) in pred for a in answers) else 0.0


def summarize_task(path):
    total = 0
    score_sum = 0.0
    legacy = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if row.get("score") is not None:
                score_sum += float(row["score"])
            else:
                legacy = True
                score_sum += _legacy_score(row)
    mean = (score_sum / total * 100.0) if total else 0.0
    return mean, total, legacy


def summarize_dir(out_dir):
    print(f"\n{'='*64}")
    print(f"RULER summary: {out_dir}")
    print(f"{'='*64}")
    print(f"{'task':<18}{'score(%)':>10}{'n':>6}  note")
    print("-" * 64)

    present = []
    means = []
    any_legacy = False
    for task in FULL_TASKS:
        path = os.path.join(out_dir, f"{task}.jsonl")
        if not os.path.exists(path):
            print(f"{task:<18}{'--':>10}{'--':>6}  MISSING")
            continue
        mean, n, legacy = summarize_task(path)
        any_legacy = any_legacy or legacy
        note = "legacy-substring" if legacy else ""
        print(f"{task:<18}{mean:>10.2f}{n:>6}  {note}")
        present.append(task)
        means.append(mean)

    print("-" * 64)
    if means:
        macro = sum(means) / len(means)
        print(f"{'MACRO-AVG':<18}{macro:>10.2f}{len(means):>6}  "
              f"over {len(present)}/{len(FULL_TASKS)} tasks"
              f"{' [LEGACY scoring present]' if any_legacy else ''}")
        if len(present) < len(FULL_TASKS):
            missing = [t for t in FULL_TASKS if t not in present]
            print(f"  [WARN] missing tasks excluded from mean: {missing}")
    else:
        print("  [WARN] no task jsonl files found")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for out_dir in sys.argv[1:]:
        summarize_dir(out_dir)


if __name__ == "__main__":
    main()
