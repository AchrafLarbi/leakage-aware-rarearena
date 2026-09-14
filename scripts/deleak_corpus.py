"""
Remove evaluation cases from the RAG corpus to eliminate data leakage.

Every case in eval_rare_cases.jsonl is drawn from RDS.json, and RDS.json case
reports are also present in rag_corpus_final.jsonl (with the diagnosis stored as
the document title). This means the evaluation cases leak into retrieval.

This script writes a clean corpus with those cases removed, keeping all other
documents (PubMed abstracts, GARD, UMLS, and the remaining RDS cases). It does
not modify the original corpus.

Run:  python scripts/deleak_corpus.py
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
EVAL = DATA / "eval_rare_cases.jsonl"
CORPUS_IN = DATA / "rag_corpus_final.jsonl"
CORPUS_OUT = DATA / "rag_corpus_clean.jsonl"


def norm(text: str) -> str:
    """Normalized key: collapse whitespace, take first 300 chars, lowercase."""
    return " ".join(str(text).split())[:300].lower()


def eval_keys():
    keys = set()
    with EVAL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            inp = r.get("input", {})
            txt = inp.get("case_report") or inp.get("case_rep") or ""
            keys.add(norm(txt))
    return keys


def main():
    keys = eval_keys()
    print(f"evaluation cases to exclude : {len(keys)}")

    kept = removed = total = 0
    with CORPUS_IN.open("r", encoding="utf-8") as fin, \
         CORPUS_OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            d = json.loads(line)
            # Only RDS case reports can collide with the eval set.
            if d.get("source_dataset") == "rds_case_reports" and norm(d.get("text", "")) in keys:
                removed += 1
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1

    print(f"corpus documents in         : {total}")
    print(f"removed (leaked eval cases) : {removed}")
    print(f"clean corpus documents      : {kept}")
    print(f"written to                  : {CORPUS_OUT.name}")

    # Verify: no eval case text remains in the clean corpus.
    remaining = 0
    with CORPUS_OUT.open("r", encoding="utf-8") as f:
        for line in f:
            if "rds_case_reports" not in line:
                continue
            d = json.loads(line)
            if norm(d.get("text", "")) in keys:
                remaining += 1
    print(f"leaked cases still present  : {remaining}  (must be 0)")


if __name__ == "__main__":
    main()
