# Data

## Files in this folder

| File | Records | Content |
|---|---:|---|
| `eval_rare_cases.jsonl` | 650 | Evaluation pool: RareArena rare-disease case reports (`input.case_report`) with the diagnosis (`target`). The notebooks sample 200 of them with seed 42. |
| `eval_evidence_labels.jsonl` | 650 | Output of `scripts/label_eval_evidence.py`: whether the de-leaked corpus still holds another case report of the same disease (`has_corpus_evidence`, 216 of 650). |
| `orphanet_disease_cards.jsonl` | 4,337 | Output of `scripts/build_orphanet_knowledge.py`: one disease-level reference card per Orphanet disorder (HPO phenotypes with frequency, prevalence class). No patient cases. |

## Large files (not in git)

GitHub rejects files over 100 MB, so the corpora are hosted on Kaggle:
<https://www.kaggle.com/datasets/larbimohammedachraf/pfe-rarediease>

| File | Size | Content |
|---|---:|---|
| `rag_corpus_final.jsonl` | ~300 MB | Contaminated retrieval corpus, 130,316 documents (PubMed abstracts, GARD, UMLS, RareArena case reports) |
| `rag_corpus_clean.jsonl` | ~300 MB | De-leaked corpus, 129,666 documents (`scripts/deleak_corpus.py`) |
| `pe_sft_final.jsonl` | ~260 MB | Few-shot exemplar pool (MedMCQA, MedQA, PubMedQA, synthetic reasoning cases; no RareArena case) |
| `rag_corpus_enriched.jsonl` | ~300 MB | De-leaked corpus + Orphanet cards, 134,003 documents (build it locally with `scripts/build_orphanet_knowledge.py`) |

Place downloaded files directly in this folder; `.gitignore` keeps them out of git.

## Orphadata inputs

`scripts/build_orphanet_knowledge.py` needs two Orphadata XML products in `data/orphadata/`:

* `en_product4_phenotypes.xml` (rare diseases with associated HPO phenotypes)
* `en_product9_epidemiology.xml` (epidemiology)

Download them from <https://www.orphadata.com>.

## Licenses and attribution

* **Orphanet / Orphadata** data are distributed under CC BY 4.0. `orphanet_disease_cards.jsonl` is
  derived from them: "Orphadata: Free access data from Orphanet. © INSERM."
* **RareArena** case reports come from PubMed Central case reports collected by RareArena. Reuse of
  `eval_rare_cases.jsonl` follows the license of the RareArena dataset.
* **UMLS** content in the retrieval corpus is subject to the UMLS Metathesaurus License, which requires
  users to hold their own UMLS license.
