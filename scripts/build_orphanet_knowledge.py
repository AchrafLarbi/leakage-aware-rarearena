"""
Build disease-level knowledge documents from Orphanet and add them to the corpus.

For every Orphanet disorder we emit ONE reference document:

    Title: <disease name>
    Text : "<name> (ORPHA:<code>) is a rare disease. Reported prevalence: <class>.
            Characteristic clinical features (HPO): <phenotype (frequency); ...>."

This is DISEASE-LEVEL knowledge (a differential-diagnosis reference), not patient
case reports. It describes the diseases in general, contains none of the evaluation
patients, and covers thousands of disorders (not just the evaluated ones), so it is
a fair reference and introduces no leakage.

Inputs (Orphanet, CC-BY-4.0):
    data/orphadata/en_product4_phenotypes.xml     disease -> HPO phenotypes (+frequency)
    data/orphadata/en_product9_epidemiology.xml   disease -> prevalence class

Outputs:
    orphanet_disease_cards.jsonl   the reference cards
    rag_corpus_enriched.jsonl      clean corpus + cards

Run: python scripts/build_orphanet_knowledge.py
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
ORPHA = DATA / "orphadata"
PHEN = ORPHA / "en_product4_phenotypes.xml"
EPI = ORPHA / "en_product9_epidemiology.xml"
CLEAN = DATA / "rag_corpus_clean.jsonl"
CARDS = DATA / "orphanet_disease_cards.jsonl"
OUT = DATA / "rag_corpus_enriched.jsonl"
EVAL = DATA / "eval_rare_cases.jsonl"

MAX_PHENOTYPES = 40
FREQ_ORDER = {"Very frequent": 0, "Frequent": 1, "Occasional": 2,
              "Very rare": 3, "Excluded": 4}


def strip_ns(tag):
    return tag.split("}")[-1]


def freq_rank(name):
    for k, v in FREQ_ORDER.items():
        if name.startswith(k):
            return v
    return 5


def parse_prevalence():
    """OrphaCode -> most specific prevalence class string."""
    prev = {}
    for _, elem in ET.iterparse(EPI, events=("end",)):
        if strip_ns(elem.tag) != "Disorder":
            continue
        code = None
        classes = []
        for ch in elem.iter():
            t = strip_ns(ch.tag)
            if t == "OrphaCode" and code is None:
                code = (ch.text or "").strip()
            if t == "PrevalenceClass":
                for sub in ch.iter():
                    if strip_ns(sub.tag) == "Name" and sub.text:
                        classes.append(sub.text.strip())
        if code:
            prev[code] = classes[0] if classes else ""
        elem.clear()
    return prev


def parse_phenotypes():
    """Yield (code, name, [(term, freq), ...])."""
    for _, elem in ET.iterparse(PHEN, events=("end",)):
        if strip_ns(elem.tag) != "Disorder":
            continue
        code = name = None
        phenos = []
        for ch in elem:
            t = strip_ns(ch.tag)
            if t == "OrphaCode":
                code = (ch.text or "").strip()
            elif t == "Name":
                name = (ch.text or "").strip()
            elif t == "HPODisorderAssociationList":
                for assoc in ch:
                    term = freq = ""
                    for sub in assoc.iter():
                        st = strip_ns(sub.tag)
                        if st == "HPOTerm":
                            term = (sub.text or "").strip()
                        elif st == "Name" and sub.text and "%" in sub.text:
                            freq = sub.text.strip()
                    if term:
                        phenos.append((term, freq))
        if code and name:
            yield code, name, phenos
        elem.clear()


def build_text(name, code, prev, phenos):
    phenos = sorted(phenos, key=lambda p: freq_rank(p[1]))[:MAX_PHENOTYPES]
    feats = "; ".join(f"{t} ({re.sub(r'[()]', '', f).strip()})" if f else t
                      for t, f in phenos)
    parts = [f"{name} (ORPHA:{code}) is a rare disease."]
    if prev:
        parts.append(f"Reported prevalence: {prev}.")
    if feats:
        parts.append(f"Characteristic clinical features (Human Phenotype Ontology): {feats}.")
    return " ".join(parts)


def main():
    print("parsing prevalence (product9)...")
    prev_map = parse_prevalence()
    print(f"  prevalence entries: {len(prev_map)}")

    print("parsing phenotypes (product4) and writing cards...")
    n = 0
    names = set()
    with CARDS.open("w", encoding="utf-8") as f:
        for code, name, phenos in parse_phenotypes():
            text = build_text(name, code, prev_map.get(code, ""), phenos)
            row = {
                "id": f"orpha:{code}",
                "source_dataset": "orphanet_disease_knowledge",
                "source_id": code,
                "title": name,
                "text": text,
                "metadata": {"orpha_code": code, "n_phenotypes": len(phenos)},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            names.add(" ".join(name.split()).lower())
            n += 1
    print(f"  disease cards written: {n}")

    # coverage: how many evaluation diseases now have a knowledge card (by name)
    covered = total = 0
    with EVAL.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            tgt = " ".join(json.loads(line).get("target", "").split()).lower()
            covered += tgt in names
    print(f"  eval diseases with a card: {covered}/{total} ({100*covered/total:.1f}%)")

    print("writing enriched corpus (clean + cards)...")
    kept = 0
    with OUT.open("w", encoding="utf-8") as fout:
        with CLEAN.open(encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    fout.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
        with CARDS.open(encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)
    print(f"  clean corpus docs : {kept}")
    print(f"  + orphanet cards  : {n}")
    print(f"  enriched total    : {kept + n}")
    print(f"  written to        : {OUT.name}")


if __name__ == "__main__":
    main()
