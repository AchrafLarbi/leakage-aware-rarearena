# Leakage-Aware Evaluation of Retrieval-Augmented Rare Disease Diagnosis

Code, data and results for the paper

> M. A. Larbi, S. Mezrar, B. Klouche, O. Pons, E. Kornyshova.
> **Leakage-Aware Evaluation of Retrieval-Augmented Rare Disease Diagnosis over Biomedical Linked Data.**
> IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT), 2026.

## What the paper shows

RareArena evaluation cases are PubMed Central case reports, and the same reports sit in a
PubMed-derived retrieval corpus **with the diagnosis stored as the document title**. A
retrieval-augmented model can therefore retrieve and copy the answer. This repository contains
everything needed to see and remove that leak:

| | Contaminated corpus | De-leaked corpus | De-leaked + Orphanet |
|---|---:|---:|---:|
| Documents | 130,316 | 129,666 | 134,003 |
| BioMistral RAG-CoT exact match (200 cases) | 59.0% | 8.5% | 4.5% |

Once the 650 evaluation cases are removed from the corpus, exact match drops about sevenfold and
most differences between retrieval and reasoning strategies stop being significant.

## Repository layout

```
notebooks/
  deleaked_biomistral_biomed.ipynb    BioMistral-7B, biomed embeddings, de-leaked corpus
  deleaked_openbiollm_fast.ipynb      Llama3-OpenBioLLM-8B, fast embeddings, de-leaked corpus
  contaminated/                       the same pipeline on the contaminated corpus
    baselines_{biomistral,openbiollm}_{biomed,fast}.ipynb   stage 4 without graph context
    advanced_biomistral_{biomed,fast}.ipynb                 GraphRAG, KAPING, RAG-CoT, CoT-RAG
    advanced_openbiollm_both.ipynb
scripts/
  deleak_corpus.py                    removes the evaluation cases from the corpus
  verify_deleak_semantic.py           8-word shingle containment check (no near-duplicates left)
  label_eval_evidence.py              Group A / Group B: does another case of the disease remain?
  build_orphanet_knowledge.py         4,337 Orphanet disease-phenotype cards + enriched corpus
  significance_from_counts.py         Fisher tests, bootstrap and difference intervals of Section IV
  answer_normalization.py             alias tables and fuzzy disease-name matching (reading copy)
  notebook_cells/
    r23_embedding_deleak_check.py     embedding check with a positive control (paste into a notebook)
    a3_sensitivity_retrieval_weights.py  12 fusion-weight settings (paste after the setup cells)
data/                                 small files; large corpora are on Kaggle (see data/README.md)
results/                              saved outputs behind the numbers in the paper
```

## Where each number in the paper comes from

| Paper element | Source |
|---|---|
| Table "Stage 4 Without Graph Context" (contaminated) | `notebooks/contaminated/baselines_*.ipynb` (best embedding profile per version) |
| Contamination gap table, contaminated column (RAG-CoT 59.0, GraphRAG 50.5, KAPING 30.5, CoT-RAG 3.5) | one BioMistral *biomed* run on `rag_corpus_final.jsonl` (see Notes) |
| Contamination gap table, de-leaked column (RAG-CoT 8.5, GraphRAG 5.5, KAPING 3.5, CoT-RAG 3.0) | `notebooks/deleaked_biomistral_biomed.ipynb`, `results/deleaked_biomistral_biomed.csv` |
| Contamination gap table, enriched column (RAG-CoT 4.5, GraphRAG 5.5, KAPING 4.0, CoT-RAG 2.0) | `results/enriched_biomistral_biomed.csv` |
| OpenBioLLM on the de-leaked corpus (best 2.5%, abstention 54.0–88.0%) | `notebooks/deleaked_openbiollm_fast.ipynb`, `results/deleaked_openbiollm_fast.csv` |
| Fisher p-values, bootstrap and difference intervals | `python scripts/significance_from_counts.py` |
| 650 documents removed, 129,666 kept | `python scripts/deleak_corpus.py` |
| Shingle containment max 0.32, mean 0.03 | `scripts/verify_deleak_semantic.py` → `results/deleak_verification.json` |
| Embedding check: 650/650 at rank 1 (contaminated), max 0.97 / mean 0.85 (de-leaked) | `scripts/notebook_cells/r23_embedding_deleak_check.py` → `results/r23_*.{json,csv}` |
| Evidence coverage 33% of evaluation cases (216/650) | `python scripts/label_eval_evidence.py` → `data/eval_evidence_labels.jsonl` |
| Retrieval recall 14.0–16.7% over 12 weightings (published 15.3%) | `scripts/notebook_cells/a3_sensitivity_retrieval_weights.py` → `results/sensitivity_retrieval_weights.json` |

## Reproducing

### 1. Data preparation (CPU, a few minutes)

```bash
pip install -r requirements.txt
# download rag_corpus_final.jsonl from Kaggle into data/ (see data/README.md)
python scripts/deleak_corpus.py            # -> data/rag_corpus_clean.jsonl
python scripts/verify_deleak_semantic.py   # -> results/deleak_verification.json
python scripts/label_eval_evidence.py       # -> data/eval_evidence_labels.jsonl
# download Orphadata products 4 and 9 into data/orphadata/ (see data/README.md)
python scripts/build_orphanet_knowledge.py # -> data/orphanet_disease_cards.jsonl, data/rag_corpus_enriched.jsonl
python scripts/significance_from_counts.py
```

### 2. Pipeline runs (GPU)

The notebooks were run on Kaggle with one 16 GB GPU; a run of the four advanced versions on
200 cases takes several hours.

1. Create a Kaggle notebook, import the `.ipynb`, and attach the dataset
   [`larbimohammedachraf/pfe-rarediease`](https://www.kaggle.com/datasets/larbimohammedachraf/pfe-rarediease).
   The notebooks read from `/kaggle/input/datasets/larbimohammedachraf/pfe-rarediease/`.
2. Add a Kaggle secret named `HF_TOKEN` (a Hugging Face read token). No token is stored in the notebooks.
3. Run all cells. The configuration cell sets the model (`BioMistral/BioMistral-7B` or
   `aaditya/Llama3-OpenBioLLM-8B`), the embedding profile (`biomed` =
   `NeuML/pubmedbert-base-embeddings`, `fast` = `sentence-transformers/all-MiniLM-L6-v2`), the corpus
   file and the 200-case sample (seed 42).
4. To run on the enriched corpus, build `rag_corpus_enriched.jsonl` with
   `scripts/build_orphanet_knowledge.py`, add it to your Kaggle dataset, and point the corpus path to it.

The embedding check and the sensitivity analysis in `scripts/notebook_cells/` are cells: paste
them at the end of a notebook, after the setup cells.

## Notes

* **De-leaking rule.** `deleak_corpus.py` removes a RareArena case report (`source_dataset ==
  "rds_case_reports"`) when its first 300 characters, after whitespace normalization and lowercasing,
  equal those of an evaluation case. The embedding check recomputes the leak mask with a 200-character
  prefix over all documents; both rules remove 650 documents (129,666 remain).
* **Answer normalization.** The advanced versions use 10 aliases mapped to 7 names plus a dictionary
  of names built from the corpus and the concept graph (41,136 names on the de-leaked corpus, 41,538 on
  the contaminated one); the baselines use 70 aliases mapped to 51 names. See `scripts/answer_normalization.py`.
* **What is not in this repository.** Per-case prediction files stayed in the Kaggle sessions, so the
  Group A / Group B split of the evidence table cannot be recomputed from the saved files, and the
  statistics use unpaired tests on counts. The enriched BioMistral run kept only its metrics file.
* **Contaminated column.** The contaminated column of the contamination gap table is one BioMistral run
  with the *biomed* profile on the same 200 cases, like the de-leaked and enriched columns. Its per-case
  file was not retained; its counts (118 / 101 / 61 / 7 correct) and the case-level comparison with the
  enriched run (109 of the 118 correct answers become wrong) were computed from it before it was
  overwritten. The EDiS notebooks in `notebooks/contaminated/` are close runs of the same pipeline
  (biomed: RAG-CoT 58.5, GraphRAG 50.5, KAPING 28.5), except that their CoT-RAG extracts phenotype terms
  before retrieval (33.0–35.0%), whereas this paper's CoT-RAG retrieves with its full chain-of-thought
  reasoning (5 questions, up to 768 tokens).
  An exploratory phenopacket section was removed from `deleaked_biomistral_biomed.ipynb`; it is
  not used in the paper.
* **Scope.** Only the retrieval corpus is decontaminated. Both models are pretrained on PubMed
  Central, so training-data contamination cannot be excluded.
* This is a research prototype evaluated with automatic metrics only. It is not clinically validated
  and must not be used for diagnosis.

## Citation

See [`CITATION.cff`](CITATION.cff).

```bibtex
@inproceedings{larbi2026leakage,
  author    = {Larbi, Mohammed Achraf and Mezrar, Samiha and Klouche, Badia and Pons, Olivier and Kornyshova, Elena},
  title     = {Leakage-Aware Evaluation of Retrieval-Augmented Rare Disease Diagnosis over Biomedical Linked Data},
  booktitle = {Proceedings of the IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT)},
  year      = {2026}
}
```

## License

Code: MIT (see [`LICENSE`](LICENSE)). Data files keep the license of their source; see
[`data/README.md`](data/README.md).
