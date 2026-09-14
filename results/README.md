# Results

All runs use the same 200 evaluation cases (seed 42). Rates in the CSV files are fractions
(0.085 = 8.5%).

| File | Produced by | Content |
|---|---|---|
| `deleaked_biomistral_biomed.csv` | `notebooks/deleaked_biomistral_biomed.ipynb` | BioMistral, biomed embeddings, de-leaked corpus: exact match, token F1, ROUGE-L, cosine, semantic accuracy, hallucination and abstention rates per version |
| `enriched_biomistral_biomed.csv` | BioMistral, biomed embeddings, on `rag_corpus_enriched.jsonl` | Same metrics on the enriched corpus (the run's notebook was not kept, only this metrics file) |
| `deleaked_openbiollm_fast.csv` | `notebooks/deleaked_openbiollm_fast.ipynb` | OpenBioLLM, fast embeddings, de-leaked corpus. Transcribed from the table printed in the notebook (6 decimals) |
| `deleak_verification.json` | `scripts/verify_deleak_semantic.py` | 8-word shingle containment of every evaluation case in the de-leaked corpus (max 0.32, mean 0.03, none above 0.5) |
| `r23_embedding_deleak_check.json` | `scripts/notebook_cells/r23_embedding_deleak_check.py` | Embedding check: contaminated corpus 650/650 own document at rank 1 (cosine 1.00); de-leaked max 0.97, mean 0.85, 3 cases above 0.95 |
| `r23_top_similar_cases.csv` | same | The 20 closest surviving (evaluation case, corpus document) pairs, inspected by hand |
| `sensitivity_retrieval_weights.json` | `scripts/notebook_cells/a3_sensitivity_retrieval_weights.py` | Recall of the correct disease in the top 5 under 12 fusion-weight settings (14.0–16.7%) |

Technique names in the CSV files: `KG-Augmented RAG` = KAPING, `RAG-driven CoT` = RAG-CoT,
`CoT-driven RAG` = CoT-RAG.
