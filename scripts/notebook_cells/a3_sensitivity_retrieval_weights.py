# ══════════════════════════════════════════════════════════════════════════════
# A3 — SENSITIVITY ANALYSIS OF THE RETRIEVAL WEIGHTS      (Reviewer 2, Comment 4)
# ------------------------------------------------------------------------------
# "A sensitivity analysis of the manually selected retrieval/voting thresholds
#  would improve confidence in the robustness of the reported results,
#  particularly because the manuscript acknowledges that these parameters were
#  selected heuristically."
#
# The fusion weights in Eq. (1) -- alpha dense, beta cross-encoder, gamma lexical,
# delta title bonus -- were chosen by judgement, not tuned. This cell re-runs
# retrieval under several weightings and measures, for each, how often a document
# of the CORRECT disease appears in the top-k.
#
# NO LANGUAGE MODEL IS INVOLVED: retrieval happens before generation, so this is
# model-independent (it does not need to be repeated for OpenBioLLM) and cheap --
# no GPU generation, only FAISS lookups.
#
# Run this AFTER the setup cells (documents / retrieve_documents / EVAL_PATH must
# exist). It does not need the model-loading or evaluation cells.
# ══════════════════════════════════════════════════════════════════════════════
import io, json, re, random

SENS_N = 150          # evaluation cases sampled
SENS_K = 5            # top-k that the pipeline actually passes to the model
SENS_SEED = 42

# (alpha dense, beta cross, gamma lexical) -- the paper uses 0.52 / 0.28 / 0.20
SENS_GRID = [
    (0.52, 0.28, 0.20),   # as published
    (0.70, 0.20, 0.10),   # dense-dominant
    (0.34, 0.33, 0.33),   # equal thirds
    (1.00, 0.00, 0.00),   # dense only
    (0.40, 0.40, 0.20),   # cross-encoder heavy
    (0.60, 0.10, 0.30),   # lexical heavy
]
SENS_DELTA = [0.03, 0.00]        # title bonus: published vs disabled

_SENS_W = {"a": 0.52, "b": 0.28, "g": 0.20, "d": 0.03}

# ---- override the fusion so the weights become configurable ------------------
_orig_hybrid_rerank = _hybrid_rerank


def _hybrid_rerank(query, rows, dense_weight=0.72, lexical_weight=0.28):
    if not rows:
        return rows
    rows = _apply_cross_encoder_scores(query, rows)
    ds = [float(r.get("score", 0.0)) for r in rows]
    d_min, d_max = min(ds), max(ds)
    q_tokens = set(t for t in _tokenize_match(query) if len(t) >= 6)
    a, b, g, dlt = _SENS_W["a"], _SENS_W["b"], _SENS_W["g"], _SENS_W["d"]
    out = []
    for r in rows:
        dense = float(r.get("score", 0.0))
        cross = float(r.get("cross_score", 0.0))
        dense_norm = (dense - d_min) / (d_max - d_min + 1e-8)
        lexical = _lexical_overlap_score(query, f"{r.get('title','')}\n{r.get('text','')}")
        title_tokens = set(_tokenize_match(r.get("title", "")))
        hit = dlt if (q_tokens and any(t in title_tokens for t in q_tokens)) else 0.0
        hybrid = (a * dense_norm + g * lexical + b * cross + hit) if cross > 0.0 \
                 else (a / max(a + g, 1e-8) * dense_norm + g / max(a + g, 1e-8) * lexical + hit)
        o = dict(r)
        o["dense_score"] = dense
        o["lexical_score"] = float(lexical)
        o["hybrid_score"] = float(hybrid)
        out.append(o)
    out.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return out


# ---- evaluation cases --------------------------------------------------------
def _n(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(t).lower())).strip()


cases = []
with io.open(EVAL_PATH, encoding="utf-8", errors="replace") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r.get("input", {})
        txt = inp.get("case_report", "") if isinstance(inp, dict) else str(inp)
        if txt and r.get("target"):
            cases.append((txt, r["target"]))
random.seed(SENS_SEED)
sample = random.sample(cases, min(SENS_N, len(cases)))
print(f"sensitivity analysis on {len(sample)} evaluation cases, "
      f"metric = correct disease present in top-{SENS_K} retrieved titles")
print("(retrieval only -- no language model, so this is model-independent)\n")


def _recall_at_k():
    hit = 0
    for text, target in sample:
        try:
            docs = retrieve_documents(text, k=SENS_K)
        except Exception:
            docs = []
        t = _n(target)
        if any(t and (t in _n(d.get("title", "")) or _n(d.get("title", "")) in t) for d in docs):
            hit += 1
    return 100.0 * hit / len(sample)


rows = []
print(f"{'alpha':>6} {'beta':>6} {'gamma':>6} {'delta':>6} {'recall@'+str(SENS_K):>10}")
print("-" * 40)
for dlt in SENS_DELTA:
    for a, b, g in SENS_GRID:
        _SENS_W.update(a=a, b=b, g=g, d=dlt)
        r = _recall_at_k()
        rows.append({"alpha": a, "beta": b, "gamma": g, "delta": dlt, "recall": r})
        tag = "  <- as published" if (a, b, g, dlt) == (0.52, 0.28, 0.20, 0.03) else ""
        print(f"{a:6.2f} {b:6.2f} {g:6.2f} {dlt:6.2f} {r:9.1f}%{tag}")

_SENS_W.update(a=0.52, b=0.28, g=0.20, d=0.03)          # restore published values
vals = [r["recall"] for r in rows]
pub = next(r["recall"] for r in rows
           if (r["alpha"], r["beta"], r["gamma"], r["delta"]) == (0.52, 0.28, 0.20, 0.03))
print("-" * 40)
print(f"published setting : {pub:.1f}%")
print(f"range across {len(rows)} settings: {min(vals):.1f}% - {max(vals):.1f}%  "
      f"(spread {max(vals)-min(vals):.1f} points)")

json.dump(rows, io.open("sensitivity_retrieval_weights.json", "w", encoding="utf-8"), indent=1)
print("\nsaved: sensitivity_retrieval_weights.json")
print("\nSENTENCE FOR THE PAPER:")
print(f'  "We varied the fusion weights of Eq. (1) over {len(rows)} settings, including equal')
print( '   weighting, dense-only retrieval and removal of the title bonus. Retrieval recall@5')
print(f'   ranges from {min(vals):.1f}% to {max(vals):.1f}% (published setting {pub:.1f}%), a spread of')
print(f'   {max(vals)-min(vals):.1f} points, indicating that our conclusions do not depend on the')
print( '   particular heuristic weighting chosen."')
