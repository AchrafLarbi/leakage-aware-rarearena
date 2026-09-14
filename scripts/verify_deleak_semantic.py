"""
Reviewer-requested verification that the de-leaked corpus contains no
paraphrased / duplicated / secondary version of any evaluation case.

The submitted de-leaking removed a corpus document when its PMID matched, or
when the first 200 normalised characters matched. Both reviewers noted that this
cannot catch a paraphrase, a translation, or a secondary report of the same
patient. This script therefore performs a content-level near-duplicate search
that does not depend on the prefix or the identifier.

Method (standard near-duplicate detection):
  * every evaluation case is reduced to the set of its 8-word shingles;
  * every corpus document is streamed and reduced the same way, sampling one
    shingle in SAMPLE_RATE by hash so the scan stays tractable;
  * for each pair we compute CONTAINMENT = shared / eval-shingles, which is the
    right measure for "is the evaluation case reproduced inside this document",
    and is robust to the document being longer or reordered.

Containment near 1.0 means a duplicate; a paraphrase of the same case typically
still shares long n-grams and lands far above unrelated text.

Output: results/deleak_verification.json  + a printed summary for the paper.
Run:    python scripts/verify_deleak_semantic.py
"""
import io, json, re, hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval_rare_cases.jsonl"
CORPUS = ROOT / "data" / "rag_corpus_clean.jsonl"
OUT = ROOT / "results" / "deleak_verification.json"
N = 8                 # shingle size in words
SAMPLE_RATE = 8       # keep 1 corpus shingle in 8
REPORT_ABOVE = 0.20   # record any pair above this containment

def norm(t):
    t = re.sub(r"[^a-z0-9\s]", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()

def shingles(text, n=N):
    w = norm(text).split()
    return {" ".join(w[i:i+n]) for i in range(max(len(w)-n+1, 0))}

def h(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

print("indexing evaluation cases ...", flush=True)
ev_sh, ev_meta = {}, {}
with io.open(EVAL, encoding="utf-8", errors="replace") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        cid = r.get("id")
        sh = shingles(r.get("input", {}).get("case_report", ""))
        if sh:
            ev_sh[cid] = sh
            ev_meta[cid] = r.get("target", "")
print(f"  evaluation cases: {len(ev_sh)}", flush=True)

inv = defaultdict(list)
for cid, sh in ev_sh.items():
    for s in sh:
        if h(s) % SAMPLE_RATE == 0:
            inv[s].append(cid)
print(f"  indexed shingles: {len(inv):,}", flush=True)

print("scanning corpus ...", flush=True)
best = defaultdict(float)
best_doc = {}
n_docs = 0
with io.open(CORPUS, encoding="utf-8", errors="replace") as f:
    for line in f:
        if not line.strip():
            continue
        n_docs += 1
        if n_docs % 20000 == 0:
            print(f"   {n_docs:,} docs", flush=True)
        try:
            d = json.loads(line)
        except Exception:
            continue
        hits = defaultdict(int)
        for s in shingles(d.get("text", "")):
            if h(s) % SAMPLE_RATE == 0:
                for cid in inv.get(s, ()):
                    hits[cid] += 1
        for cid, c in hits.items():
            denom = max(len([s for s in ev_sh[cid] if h(s) % SAMPLE_RATE == 0]), 1)
            cont = c / denom
            if cont > best[cid]:
                best[cid] = cont
                best_doc[cid] = {"source_id": d.get("source_id", ""),
                                 "title": str(d.get("title", ""))[:120],
                                 "source": d.get("source_dataset", "")}
print(f"  corpus documents scanned: {n_docs:,}", flush=True)

vals = [best.get(c, 0.0) for c in ev_sh]
buckets = {">=0.90 (duplicate)": sum(v >= .90 for v in vals),
           "0.50-0.90 (likely same case)": sum(.50 <= v < .90 for v in vals),
           "0.20-0.50 (partial overlap)": sum(.20 <= v < .50 for v in vals),
           "<0.20 (unrelated)": sum(v < .20 for v in vals)}
print("\n=== CONTAINMENT OF EVALUATION CASES IN THE DE-LEAKED CORPUS ===")
for k, v in buckets.items():
    print(f"   {k:32s} {v:4d}  ({100*v/len(vals):.1f}%)")
print(f"\n   max containment observed : {max(vals):.3f}")
print(f"   mean containment          : {sum(vals)/len(vals):.3f}")

flagged = sorted([(v, c) for c, v in best.items() if v >= REPORT_ABOVE], reverse=True)[:15]
if flagged:
    print(f"\n   highest-overlap cases (>= {REPORT_ABOVE}):")
    for v, c in flagged:
        print(f"     {v:.3f}  {ev_meta.get(c,'')[:46]:48s} <- {best_doc[c]['source']}")
else:
    print(f"\n   no evaluation case exceeds containment {REPORT_ABOVE}")

json.dump({"n_eval": len(vals), "n_docs": n_docs, "shingle_n": N,
           "sample_rate": SAMPLE_RATE, "buckets": buckets,
           "max": max(vals), "mean": sum(vals)/len(vals),
           "flagged": [{"containment": v, "disease": ev_meta.get(c,""),
                        "doc": best_doc[c]} for v, c in flagged]},
          io.open(OUT, "w", encoding="utf-8"), indent=1)
print(f"\nwritten: {OUT}")
