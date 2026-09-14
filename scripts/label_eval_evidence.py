"""
Tag every evaluation case by whether the CLEAN corpus contains legitimate
evidence for it: a different case (different text) of the same disease.

  Group A (has_corpus_evidence = True):  another case of the same disease exists
                                         in the corpus -> real retrieval possible
  Group B (has_corpus_evidence = False): the disease appears only as the test
                                         case -> no legitimate evidence exists

This is NOT leakage: the exact test case was removed by deleak_corpus.py. Here we
only measure, per evaluation case, whether a *different* same-disease case remains.

Output: eval_evidence_labels.jsonl  (one row per eval case)
Run:    python scripts/label_eval_evidence.py
"""

import json
from pathlib import Path
from collections import defaultdict

DATA = Path(__file__).resolve().parent.parent / "data"
EVAL = DATA / "eval_rare_cases.jsonl"
CORPUS = DATA / "rag_corpus_clean.jsonl"
OUT = DATA / "eval_evidence_labels.jsonl"


def dnorm(t):
    return " ".join(str(t).split()).lower().strip()


def tnorm(t):
    return " ".join(str(t).split())[:200].lower()


def main():
    # corpus: diagnosis -> set of case-text keys (only RDS case reports)
    by_diag = defaultdict(set)
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            if "rds_case_reports" not in line:
                continue
            d = json.loads(line)
            by_diag[dnorm(d.get("title", ""))].add(tnorm(d.get("text", "")))

    rows_out = []
    a = b = 0
    with EVAL.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            diag = dnorm(r.get("target", ""))
            txt = tnorm(r.get("input", {}).get("case_report", ""))
            # same-disease corpus cases whose text differs from this eval case
            others = by_diag.get(diag, set()) - {txt}
            has = len(others) > 0
            a += has
            b += (not has)
            rows_out.append({
                "id": r.get("id"),
                "target": r.get("target"),
                "has_corpus_evidence": has,
                "n_same_disease_cases_in_corpus": len(others),
            })

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows_out)
    print(f"evaluation cases labelled : {total}")
    print(f"  Group A (has evidence)  : {a}  ({100*a/total:.1f}%)")
    print(f"  Group B (no evidence)   : {b}  ({100*b/total:.1f}%)")
    print(f"written to                : {OUT.name}")
    print()
    print("Join eval_evidence_labels.jsonl to your per-case predictions by 'id'")
    print("to report accuracy on Group A vs Group B.")


if __name__ == "__main__":
    main()
