"""
Statistics reported in Section IV of the paper, recomputed from correct-answer
counts on the 200 evaluated cases.

Only aggregate counts were kept for these runs (see results/*.csv), so every
comparison uses the unpaired Fisher exact test, which is conservative relative
to the paired McNemar test.

  * Fisher exact test (two-sided) between two versions
  * 95% percentile bootstrap interval of one exact-match rate
  * 95% Wald interval of the difference between two rates

Run: python scripts/significance_from_counts.py
"""

import random
from itertools import combinations
from math import comb, sqrt

N = 200          # evaluated cases per run
BOOT = 10000
random.seed(0)


def fisher(a, b, n=N):
    """Two-sided Fisher exact p for a/n vs b/n correct answers."""
    k, total = a + b, 2 * n
    prob = lambda x: comb(n, x) * comb(n, k - x) / comb(total, k)
    p_obs = prob(a)
    return sum(prob(x) for x in range(max(0, k - n), min(n, k) + 1)
               if prob(x) <= p_obs * (1 + 1e-9))


def bootstrap_ci(correct, n=N):
    vals = [1] * correct + [0] * (n - correct)
    means = sorted(sum(random.choice(vals) for _ in range(n)) / n for _ in range(BOOT))
    return 100 * means[int(0.025 * BOOT)], 100 * means[int(0.975 * BOOT)]


def diff_ci(a, b, n=N):
    p1, p2 = a / n, b / n
    se = sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    d = p1 - p2
    return 100 * (d - 1.96 * se), 100 * (d + 1.96 * se)


# correct answers out of 200 (exact match x 200)
CONTAMINATED = {"RAG-CoT": 118, "CoT-RAG": 70, "GraphRAG": 101, "KAPING": 57}  # notebooks/contaminated/advanced_*.ipynb
DELEAKED     = {"RAG-CoT": 17, "GraphRAG": 11, "KAPING": 7, "CoT-RAG": 6}      # results/deleaked_biomistral_biomed.csv
ENRICHED     = {"RAG-CoT": 9, "GraphRAG": 11, "KAPING": 8, "CoT-RAG": 4}       # results/enriched_biomistral_biomed.csv
OPEN_FAST    = {"GraphRAG": 5, "KAPING": 4, "RAG-CoT": 4, "CoT-RAG": 3}        # results/deleaked_openbiollm_fast.csv


def main():
    print("Section IV-C  contamination manufactures findings (BioMistral)")
    for x, y in [("RAG-CoT", "CoT-RAG"), ("GraphRAG", "KAPING")]:
        print(f"  contaminated {x} vs {y}: {CONTAMINATED[x]} vs {CONTAMINATED[y]}, "
              f"p = {fisher(CONTAMINATED[x], CONTAMINATED[y]):.2e}")
    for x, y in [("GraphRAG", "KAPING"), ("RAG-CoT", "GraphRAG"), ("RAG-CoT", "CoT-RAG")]:
        print(f"  de-leaked    {x} vs {y}: {DELEAKED[x]} vs {DELEAKED[y]}, "
              f"p = {fisher(DELEAKED[x], DELEAKED[y]):.2f}")
    for v in ("RAG-CoT", "GraphRAG"):
        lo, hi = bootstrap_ci(DELEAKED[v])
        print(f"  bootstrap 95% CI {v}: {100 * DELEAKED[v] / N:.1f}% [{lo:.1f}, {hi:.1f}]")
    p_open = min(fisher(OPEN_FAST[x], OPEN_FAST[y]) for x, y in combinations(OPEN_FAST, 2))
    print(f"  OpenBioLLM fast, de-leaked: best {100 * max(OPEN_FAST.values()) / N:.1f}%, "
          f"smallest pairwise p = {p_open:.2f}")

    print("\nSection IV-D  enrichment (RAG-CoT, de-leaked vs enriched)")
    a, b = DELEAKED["RAG-CoT"], ENRICHED["RAG-CoT"]
    lo, hi = diff_ci(a, b)
    print(f"  {a}/200 vs {b}/200: difference {100 * (a - b) / N:.1f} points, "
          f"95% CI [{lo:+.1f}, {hi:+.1f}], Fisher p = {fisher(a, b):.2f}")


if __name__ == "__main__":
    main()
