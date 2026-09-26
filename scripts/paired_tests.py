"""
Paired analysis of the contaminated and de-leaked runs (Section IV-B and IV-C of the paper).

Both runs cover the same 200 cases with the same code, so every comparison here is paired:

  * how many correct answers survive once the evaluation cases leave the corpus
  * McNemar's exact test between two versions inside one corpus
  * McNemar's exact test for one version across the two corpora
  * 95% percentile bootstrap intervals on the de-leaked corpus

Inputs (per-case files written by the notebooks):
    results/contaminated_biomistral_biomed_per_case.csv
    results/deleaked_biomistral_biomed_per_case.csv

Run: python scripts/paired_tests.py
"""

import csv
import io
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTAMINATED = ROOT / "results" / "contaminated_biomistral_biomed_per_case.csv"
DELEAKED = ROOT / "results" / "deleaked_biomistral_biomed_per_case.csv"
TECHNIQUES = ["RAG-driven CoT", "GraphRAG", "KG-Augmented RAG", "CoT-driven RAG"]
BOOT = 10000
random.seed(0)


def load(path):
    """technique -> {case id: 1 if exact match else 0}"""
    per = defaultdict(dict)
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                em = 1 if float(r["exact_match"]) >= 0.5 else 0
            except (TypeError, ValueError):
                em = 0
            per[r["technique"]][r["idx"]] = em
    return per


def mcnemar(a, b):
    """Two-sided exact McNemar on the shared cases. Returns (only-a, only-b, p)."""
    ids = sorted(set(a) & set(b))
    b_only = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
    c_only = sum(1 for i in ids if a[i] == 0 and b[i] == 1)
    n, k = b_only + c_only, min(b_only, c_only)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)) if n else 1.0
    return b_only, c_only, p


def bootstrap_ci(values):
    n = len(values)
    means = sorted(sum(values[random.randrange(n)] for _ in range(n)) / n for _ in range(BOOT))
    return 100 * means[int(0.025 * BOOT)], 100 * means[int(0.975 * BOOT)]


def main():
    cont, clean = load(CONTAMINATED), load(DELEAKED)

    print("Case-level effect of removing the evaluation cases (same 200 cases)")
    print(f"  {'version':18s} {'contam':>7} {'clean':>7} {'lost':>6} {'kept':>6} {'gained':>7}")
    for t in TECHNIQUES:
        ids = sorted(set(cont[t]) & set(clean[t]))
        ce = sum(cont[t][i] for i in ids)
        le = sum(clean[t][i] for i in ids)
        lost = sum(1 for i in ids if cont[t][i] == 1 and clean[t][i] == 0)
        kept = sum(1 for i in ids if cont[t][i] == 1 and clean[t][i] == 1)
        gained = sum(1 for i in ids if cont[t][i] == 0 and clean[t][i] == 1)
        print(f"  {t:18s} {100*ce/len(ids):6.1f}% {100*le/len(ids):6.1f}% {lost:6d} {kept:6d} {gained:7d}"
              f"   ({lost}/{ce} of the correct answers lost)" if ce else "")

    print("\nMcNemar exact test between versions")
    pairs = [("RAG-driven CoT", "CoT-driven RAG"), ("GraphRAG", "KG-Augmented RAG"),
             ("RAG-driven CoT", "GraphRAG")]
    for label, per in [("contaminated", cont), ("de-leaked", clean)]:
        for x, y in pairs:
            _, _, p = mcnemar(per[x], per[y])
            print(f"  {label:12s} {x:16s} vs {y:18s} p = {p:.3g}")

    print("\nMcNemar exact test, same version across the two corpora")
    for t in TECHNIQUES:
        lost, gained, p = mcnemar(cont[t], clean[t])
        print(f"  {t:18s} lost {lost:3d}, gained {gained:2d}, p = {p:.3g}")

    print("\n95% bootstrap intervals on the de-leaked corpus")
    for t in TECHNIQUES:
        vals = list(clean[t].values())
        lo, hi = bootstrap_ci(vals)
        print(f"  {t:18s} EM = {100*sum(vals)/len(vals):4.1f}%  [{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    main()
