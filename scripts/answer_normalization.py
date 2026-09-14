"""
Disease-name normalization used to score every answer (paper, Section III-G).

Two variants exist, both extracted VERBATIM from the notebooks so the alias
tables and matching rules can be read without opening a notebook:

A) Advanced versions (GraphRAG, KAPING, RAG-CoT, CoT-RAG), i.e. every
   de-leaked / enriched result and the contaminated advanced column.
   Source: notebooks/deleaked_biomistral_biomed.ipynb, cells 7 and 17
   (identical code in the other advanced notebooks).
     * _norm_disease_name       lowercase, Unicode NFKD -> ASCII, strip punctuation
     * DISEASE_ALIAS_MAP        10 hand-written surface forms -> 7 canonical names
     * _best_dictionary_match   token Jaccard + 0.25 containment bonus against
                                DISEASE_CANONICAL_LOOKUP, built from concept-graph
                                nodes, corpus titles and OMIM/HPO metadata
                                (41,136 names on the de-leaked corpus, 41,538 on the
                                contaminated corpus); accepted if score >= 0.58, or
                                >= 0.48 with at least 2 shared tokens

B) Baselines without graph context (paper table "Stage 4 Without Graph Context").
   Source: notebooks/contaminated/baselines_*.ipynb, cell 16.
     * DISEASE_ALIASES          70 surface forms -> 51 canonical names; these 51
                                names are also the fuzzy-matching dictionary
     * same Jaccard + containment rule and thresholds

This file is a reading copy: the functions depend on objects created earlier in
each notebook (normalize_text, extract_final_diagnosis, kg_node_names, df_rag),
and the two variants define the same names, so do not import it as a module.
"""

# ============================================================================
# A) ADVANCED VERSIONS
# ============================================================================

# ----- notebook cell 7 (excerpt) -----
# ── Disease dictionary (for post-generation canonicalization) ──────────────────
def _norm_disease_name(text: str) -> str:
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"'", "", text)
    text = re.sub(r"[-\u2013\u2014]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_plausible_disease_name(name: str) -> bool:
    n = _norm_disease_name(name)
    if len(n) < 4:
        return False
    if n in {"rare diseases", "orphan diseases", "hereditary diseases"}:
        return False
    # Filter ultra-generic one-token noise from metadata lists.
    if len(n.split()) == 1 and n in {
        "pain", "mild", "left", "right", "fever", "coma", "falls", "frequent", "focal", "diffuse"
    }:
        return False
    return True


def _collect_rag_disease_candidates(df: pd.DataFrame) -> list:
    candidates = []
    if df.empty:
        return candidates

    if "title" in df.columns:
        candidates.extend(df["title"].dropna().astype(str).tolist())

    if "metadata" in df.columns:
        for meta in df["metadata"].dropna().tolist():
            if not isinstance(meta, dict):
                continue
            for key in ("omim_titles", "hpo_names"):
                vals = meta.get(key, [])
                if isinstance(vals, list):
                    candidates.extend([str(v) for v in vals if isinstance(v, (str, int, float))])
    return candidates


KG_DISEASE_NAMES = [name for _, name in kg_node_names]
RAG_DISEASE_CANDIDATES = _collect_rag_disease_candidates(df_rag)

_all_candidates = KG_DISEASE_NAMES + RAG_DISEASE_CANDIDATES
_all_candidates = [str(c).strip() for c in _all_candidates if _is_plausible_disease_name(c)]

DISEASE_CANONICAL_LOOKUP = {}
for c in sorted(set(_all_candidates), key=lambda x: (len(x), x.lower())):
    n = _norm_disease_name(c)
    if n and n not in DISEASE_CANONICAL_LOOKUP:
        DISEASE_CANONICAL_LOOKUP[n] = c
DISEASE_CANONICAL_SET = set(DISEASE_CANONICAL_LOOKUP.keys())

print(f"Disease dictionary size (normalized): {len(DISEASE_CANONICAL_SET)}")


# ----- notebook cell 17 (excerpt) -----
# ─── Disease canonicalization (dictionary + aliases) ──────────────────────────
DISEASE_ALIAS_MAP = {
    "anti nmdar encephalitis": "anti nmda receptor encephalitis",
    "anti nmda receptor encephalitis": "anti nmda receptor encephalitis",
    "x linked agammaglobulinemia": "x linked agammaglobulinemia",
    "xla": "x linked agammaglobulinemia",
    "miller fisher syndrome": "miller fisher syndrome",
    "kikuchi fujimoto disease": "kikuchi fujimoto disease",
    "kikuchi disease": "kikuchi fujimoto disease",
    "wilson disease": "wilson disease",
    "diffuse large b cell lymphoma": "diffuse large b cell lymphoma",
    "intravascular large b cell lymphoma": "intravascular large b cell lymphoma",
}


def _preclean_disease_phrase(text: str) -> str:
    t = normalize_text(text)
    t = re.sub(r"\b(the|a|an)\b", " ", t)
    t = re.sub(r"\b(single|most|likely|probable|possible|final|diagnosis|is|was|are|be|to|of|for|this|that|patient|has|have|with)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


DISEASE_TOKEN_INDEX = {}
for n in DISEASE_CANONICAL_SET:
    for tok in set(n.split()):
        if len(tok) >= 4:
            DISEASE_TOKEN_INDEX.setdefault(tok, set()).add(n)


def _best_dictionary_match(candidate_norm: str):
    if not candidate_norm:
        return None

    # Exact canonical hit.
    if candidate_norm in DISEASE_CANONICAL_SET:
        return candidate_norm

    toks = set(candidate_norm.split())
    if not toks:
        return None

    pool = set()
    for tok in toks:
        if len(tok) >= 4:
            pool |= DISEASE_TOKEN_INDEX.get(tok, set())
    if not pool:
        return None

    best_name, best_score = None, 0.0
    for n in pool:
        nt = set(n.split())
        inter = len(toks & nt)
        if inter == 0:
            continue
        j = inter / max(len(toks | nt), 1)
        contain = 0.25 if (n in candidate_norm or candidate_norm in n) else 0.0
        score = j + contain
        if score > best_score:
            best_name, best_score = n, score

    if best_name is None:
        return None
    if best_score >= 0.58:
        return best_name
    if best_score >= 0.48 and len(set(best_name.split()) & toks) >= 2:
        return best_name
    return None


def canonicalize_disease_name(text: str) -> str:
    raw = str(text).strip()
    if not raw:
        return ""

    base = _preclean_disease_phrase(raw)
    if not base:
        return ""

    # Alias canonicalization first.
    if base in DISEASE_ALIAS_MAP:
        alias_norm = normalize_text(DISEASE_ALIAS_MAP[base])
        return DISEASE_CANONICAL_LOOKUP.get(alias_norm, DISEASE_ALIAS_MAP[base])

    # Dictionary match.
    best = _best_dictionary_match(base)
    if best:
        return DISEASE_CANONICAL_LOOKUP.get(best, best)

    # Fallback to cleaned text.
    return base


def get_canonical_prediction(prediction: str) -> str:
    return canonicalize_disease_name(extract_final_diagnosis(prediction))


def get_canonical_target(target: str) -> str:
    return canonicalize_disease_name(target)


# ============================================================================
# B) BASELINES WITHOUT GRAPH CONTEXT
# ============================================================================

# ----- notebook cell 16 (excerpt) -----
# ─── Disease Name Canonicalization ────────────────────────────────────────────
# Ported from advanced notebook — maps abbreviations, synonyms, and common
# variants to a single canonical disease name for fairer EM evaluation.

DISEASE_ALIASES = {
    # Abbreviation → Full Name
    "dlbcl": "diffuse large b cell lymphoma",
    "sle": "systemic lupus erythematosus",
    "all": "acute lymphoblastic leukemia",
    "aml": "acute myeloid leukemia",
    "cml": "chronic myeloid leukemia",
    "cll": "chronic lymphocytic leukemia",
    "hus": "hemolytic uremic syndrome",
    "tma": "thrombotic microangiopathy",
    "gbs": "guillain barre syndrome",
    "mfs": "miller fisher syndrome",
    "sma": "spinal muscular atrophy",
    "als": "amyotrophic lateral sclerosis",
    "pnh": "paroxysmal nocturnal hemoglobinuria",
    "ttp": "thrombotic thrombocytopenic purpura",
    "hsp": "henoch schonlein purpura",
    "iga vasculitis": "iga vasculitis henoch schonlein purpura",
    "nmo": "neuromyelitis optica",
    "nmosd": "neuromyelitis optica spectrum disorder",
    "pbc": "primary biliary cholangitis",
    "psc": "primary sclerosing cholangitis",
    "anca vasculitis": "anca associated vasculitis",
    "gpa": "granulomatosis with polyangiitis",
    "mpa": "microscopic polyangiitis",
    "egpa": "eosinophilic granulomatosis with polyangiitis",
    "ecd": "erdheim chester disease",
    "lch": "langerhans cell histiocytosis",
    "hlh": "hemophagocytic lymphohistiocytosis",
    "mas": "macrophage activation syndrome",
    "igg4 rd": "igg4 related disease",
    "igg4 related disease": "igg4 related disease",
    "igg4 rd": "igg4 related disease",
    "behcets": "behcet disease",
    "behcets disease": "behcet disease",
    "behcet syndrome": "behcet disease",
    "schnitzlers syndrome": "schnitzler syndrome",
    "schnitzlers": "schnitzler syndrome",
    "stills disease": "adult onset still disease",
    "adult onset stills disease": "adult onset still disease",
    "castlemans disease": "castleman disease",
    "wilsons disease": "wilson disease",
    "addisons disease": "addison disease",
    "cushings syndrome": "cushing syndrome",
    "pagets disease": "paget disease",
    "fabrys disease": "fabry disease",
    "fabry disease": "fabry disease",
    "gauchers disease": "gaucher disease",
    "gaucher disease": "gaucher disease",
    "hodgkins disease": "hodgkin lymphoma",
    "hodgkins lymphoma": "hodgkin lymphoma",
    "non hodgkins lymphoma": "non hodgkin lymphoma",
    "burgers disease": "buerger disease",
    "buergers disease": "buerger disease",
    "kikuchi fujimoto": "kikuchi fujimoto disease",
    "kikuchi disease": "kikuchi fujimoto disease",
    "takayasus arteritis": "takayasu arteritis",
    "wegeners granulomatosis": "granulomatosis with polyangiitis",
    "churg strauss": "eosinophilic granulomatosis with polyangiitis",
    "von hippel lindau": "von hippel lindau disease",
    "vhl": "von hippel lindau disease",
    "tuberous sclerosis": "tuberous sclerosis complex",
    "tsc": "tuberous sclerosis complex",
    "marfans syndrome": "marfan syndrome",
    "marfan": "marfan syndrome",
    "ehlers danlos": "ehlers danlos syndrome",
    "eds": "ehlers danlos syndrome",
    "pheochromocytoma": "pheochromocytoma",
    "pheo": "pheochromocytoma",
    "paraganglioma": "paraganglioma",
    "myasthenia gravis": "myasthenia gravis",
    "mg": "myasthenia gravis",
    "lupus": "systemic lupus erythematosus",
}

# Build a normalized lookup: normalize_text(alias) → canonical
DISEASE_ALIAS_MAP = {}
for alias, canonical in DISEASE_ALIASES.items():
    DISEASE_ALIAS_MAP[normalize_text(alias)] = normalize_text(canonical)

# Build canonical set for fuzzy matching
DISEASE_CANONICAL_NAMES = sorted(set(DISEASE_ALIAS_MAP.values()))
DISEASE_CANONICAL_LOOKUP = {n: n for n in DISEASE_CANONICAL_NAMES}


def _preclean_disease_phrase(text: str) -> str:
    """Remove common prefixes/suffixes that are not part of the disease name."""
    t = normalize_text(text)
    # Strip common prefixes
    for prefix in ['the diagnosis is', 'the most likely diagnosis is',
                   'final diagnosis', 'diagnosis', 'the patient has',
                   'the patient likely has', 'most likely',
                   'based on the clinical presentation']:
        if t.startswith(prefix):
            t = t[len(prefix):].strip().lstrip(':').strip()
    return t


def _best_dictionary_match(candidate: str):
    """Jaccard fuzzy match against canonical disease names."""
    candidate_norm = normalize_text(candidate)
    if not candidate_norm:
        return None

    toks = set(candidate_norm.split())
    if len(toks) < 1:
        return None

    best_name = None
    best_score = -1.0

    for n in DISEASE_CANONICAL_NAMES:
        ntoks = set(n.split())
        inter = len(toks & ntoks)
        union = len(toks | ntoks)
        if union == 0:
            continue
        j = inter / union
        # Bonus for substring containment
        contain = 0.25 if (n in candidate_norm or candidate_norm in n) else 0.0
        score = j + contain
        if score > best_score:
            best_name, best_score = n, score

    if best_name is None:
        return None
    if best_score >= 0.58:
        return best_name
    if best_score >= 0.48 and len(set(best_name.split()) & toks) >= 2:
        return best_name
    return None


def canonicalize_disease_name(text: str) -> str:
    """Canonicalize a disease name via alias lookup + fuzzy dictionary match."""
    raw = str(text).strip()
    if not raw:
        return ""

    base = _preclean_disease_phrase(raw)
    if not base:
        return ""

    # Alias canonicalization first
    if base in DISEASE_ALIAS_MAP:
        alias_norm = normalize_text(DISEASE_ALIAS_MAP[base])
        return DISEASE_CANONICAL_LOOKUP.get(alias_norm, DISEASE_ALIAS_MAP[base])

    # Dictionary match
    best = _best_dictionary_match(base)
    if best:
        return DISEASE_CANONICAL_LOOKUP.get(best, best)

    # Fallback to cleaned text
    return base


def get_canonical_prediction(prediction: str) -> str:
    """Extract and canonicalize the predicted disease name."""
    return canonicalize_disease_name(extract_final_diagnosis(prediction))


def get_canonical_target(target: str) -> str:
    """Canonicalize the ground truth target."""
    return canonicalize_disease_name(target)


# ─── Extract Final Diagnosis ──────────────────────────────────────────────────
def extract_final_diagnosis(text):
    """
    Extract the disease name from the model's response.
    Priority order:
      1. 'Final diagnosis: <name>' pattern
      2. 'the most likely diagnosis is <name>' pattern
      3. 'the diagnosis is <name>' pattern
      4. If short response (<100 chars), return entire response cleaned
    """
    text = str(text).strip()
    if not text:
        return ''

    # Pattern 1: "Final diagnosis: ..."
    m = re.search(r'(?i)final\s+diagnosis\s*[:\-–—]\s*(.+)', text)
    if m:
        diag = m.group(1).strip()
        diag = re.split(r'[\n\r]', diag)[0].strip()
        diag = re.sub(r'\s*\(?\s*(Explanation|Reasoning|Note|Because|Step|Confidence|References?)\b.*', '', diag, flags=re.IGNORECASE)
        diag = diag.strip(' .')
        if len(diag) >= 2:
            return diag

    # Pattern 2: "the most likely diagnosis is ..."
    m = re.search(r'(?i)(?:the\s+)?most\s+likely\s+diagnosis\s+is\s+(.+?)[\.\n]', text)
    if m:
        return m.group(1).strip(' .')

    # Pattern 3: "the diagnosis is ..."
    m = re.search(r'(?i)(?:the\s+)?diagnosis\s+is\s+(?:consistent\s+with\s+)?(.+?)[\.\n]', text)
    if m:
        return m.group(1).strip(' .')

    # Short response fallback
    if len(text) < 100:
        cleaned = text.strip()
        cleaned = re.sub(r'^(?:the\s+)?(?:most\s+likely\s+)?(?:diagnosis\s+is\s+)?', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip(' .')

    # Last resort: last sentence of the response
    sentences = re.split(r'[.\n]', text)
    for s in reversed(sentences):
        s = s.strip()
        if len(s) >= 3:
            return s
    return text[:100]


# ─── Lexical / Semantic Metrics (with canonicalization) ───────────────────────
