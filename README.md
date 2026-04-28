# CoveR

**CoveR** is the codebase for _"Search for Coverage: Learning Coverage-Aware Retrieval with Augmented Sub-Question Answerability"_.

The goal is to train dense retrieval models that maximise **coverage** — the fraction of answerable sub-topics in a multi-aspect information need that the retrieved documents collectively address — rather than simply optimising for per-query relevance.

---

## Overview

Standard dense retrievers are trained on relevance-annotated pairs (e.g., MS-MARCO). For complex, research-style requests that span multiple sub-aspects, a document may be relevant to the request as a whole but only partially address its sub-topics. CoveR trains on coverage-labelled data derived from [CRUX-Researchy](https://huggingface.co/datasets/DylanJHJ/crux-researchy), where each document is judged on how many of a request's sub-questions it can answer.

<!-- Key ideas: -->
<!-- - **Coverage score**: for a document _d_ and request _q_ with _k_ answerable sub-topics, coverage is the fraction of sub-topics for which _d_ receives a judge rating ≥ τ. -->
<!-- - **Coverage-based sampling**: positive/negative pairs are constructed from coverage buckets (high ≥ 0.75, half 0.5–0.75, quarter 0.25–0.5, low 0–0.25, zero = 0) rather than by rank alone. -->
<!-- - **Sub-question augmentation**: the request is decomposed into sub-queries, which are used as additional query views during training to align document representations with fine-grained information needs. -->
<!-- - **CovDistil**: a coverage-distillation loss that transfers a teacher's coverage distribution (e.g., from a reranker) to the student retriever. -->

<!-- ## Repository structure -->
<!--  -->
<!-- ``` -->
<!-- CoveR/ -->
<!-- ├── tevatron/              # Modified Tevatron framework (training + encoding + search) -->
<!-- ├── src/ -->
<!-- │   ├── multi-view-data-curation/      # Coverage-based positive/negative sampling -->
<!-- │   ├── single-view-data-curation/     # Rank-based and KD-based sampling (MS-MARCO) -->
<!-- │   ├── evaluation-data-curation/      # Scripts to build BEIR / NeuCLIR / CRUX eval sets -->
<!-- │   ├── create-crux-mds-subqueries/    # Sub-question generation from research requests -->
<!-- │   ├── pretokenization/               # Pre-tokenisation utilities -->
<!-- │   ├── sig-test/                      # Paired t-test significance testing -->
<!-- │   └── coverage-analysis/             # Coverage distribution analysis -->
<!-- ├── slurm/ -->
<!-- │   ├── modernbert/                    # ModernBERT MS-MARCO pre-fine-tuning -->
<!-- │   ├── modernbert.crux-researchy/     # ModernBERT CoveR training (coverage sampling) -->
<!-- │   ├── exp-for-init/                  # Two-query CoveR training (request + sub-queries) -->
<!-- │   ├── exp-for-ft/                    # Further fine-tuning experiments -->
<!-- │   ├── training/                      # CE+contrastive training variants -->
<!-- │   └── use-query/                     # Use-query ablations -->
<!-- ├── eval.beir/             # BEIR encoding + search scripts -->
<!-- ├── eval.beir-subset/      # BEIR subset (with baselines) -->
<!-- ├── eval.crux/             # CRUX-MDS evaluation scripts -->
<!-- ├── eval.msmarco-passage/  # MS-MARCO passage evaluation scripts -->
<!-- ├── eval.neuclir/          # NeuCLIR evaluation scripts -->
<!-- ├── eval-on-lumi/          # Lumi-cluster evaluation scripts (ModernBERT + Qwen3) -->
<!-- ├── configs/               # DeepSpeed / accelerate configs -->
<!-- └── legacy/                # Archived experiments -->
<!-- ``` -->

---

## Training pipeline

### 1. Pre-fine-tuning (PFT) — optional

Fine-tune a base model on MS-MARCO passage retrieval as a warm-start before coverage training.

```bash
# Standard contrastive training on MS-MARCO
sbatch slurm/modernbert/train.modernbert.sh          # relevance-based
sbatch slurm/modernbert/train.modernbert.kd.sh       # with KD scores from Qwen3-0.6B reranker
```

The PFT checkpoint (`DylanJHJ/nomic.modernbert-base.msmarco-passage.10k` or similar) is used as the starting point for coverage training.

### 2. Coverage-based training (CoveR)

Train on CRUX-Researchy using coverage-bucket sampling pairs.

```bash
# Run all 7 coverage-sampling ablations as a SLURM array job
sbatch slurm/modernbert.crux-researchy/train.modernbert.cov-sampling.sh

# Single run with the default split (pos_20.neg_51.filtered)
sbatch slurm/modernbert.crux-researchy/train.modernbert.sh
```

Coverage splits:

| Split | Positives | Negatives |
|---|---|---|
| `pos_20.neg_51` | top-20 by rank | rank 51+ |
| `pos_20.neg_51.filtered` | top-20, zero-coverage excluded | rank 51+ |
| `pos_high.neg_zero` | cov ≥ 0.75 | cov = 0 |
| `pos_high.neg_low` | cov ≥ 0.75 | 0 ≤ cov < 0.25 |
| `pos_high.neg_quarter` | cov ≥ 0.75 | 0 ≤ cov < 0.5 |
| `pos_half.neg_zero` | 0.5 ≤ cov < 0.75 | cov = 0 |
| `pos_low.neg_zero` | 0 ≤ cov < 0.25 | cov = 0 |

### 3. Sub-question augmented training (CoveR + SQ)

Uses the full research request **and** decomposed sub-queries as dual query views (`--request_as_query`, `--subquery_prefix`). Also supports a CovDistil KLD loss (`--covdistil_lambda`).

```bash
sbatch slurm/exp-for-init/unsupervised-pft.scope-ft.sh   # unsupervised PFT → CoveR+SQ
sbatch slurm/exp-for-init/msmarco-pft.scope-ft.sh        # MS-MARCO PFT → CoveR+SQ
```

### 4. Two-stage training

First fine-tune on flat (relevance) CRUX-Researchy data, then continue on coverage-based data.

```bash
sbatch slurm/modernbert.crux-researchy/train.modernbert.two-stage.sh
```

---

## Data curation

### Coverage-based sampling (multi-view)

```bash
# Build coverage-bucketed training sets from CRUX-Researchy judgments
python src/multi-view-data-curation/kd-sample-from-coverage.py --split train --tau 4
python src/multi-view-data-curation/sample-for-kdcovdistil.py
python src/multi-view-data-curation/sample-for-selfcovdistil.py
```

### Sub-question generation

```bash
# Generate sub-questions from research requests using Qwen2.5-7B-Instruct
python src/create-crux-mds-subqueries/generate_subquestions_from_request.py
python src/create-crux-mds-subqueries/generate_subquestions_from_request_neuclir.py
```

### Rank-based sampling (single-view, MS-MARCO)

```bash
python src/single-view-data-curation/kd-sample-from-bm25.py  # BM25 + Qwen3 KD scores
python src/single-view-data-curation/rerank-for-msmarco.py   # rerank for score labels
python src/single-view-data-curation/flatten-relevance.py    # flatten for single-view
```

### Evaluation data curation

```bash
python src/evaluation-data-curation/create_beir_corpus.py
python src/evaluation-data-curation/create_msmarco-passage_data.py
python src/evaluation-data-curation/create_nano_beir_data.py
```

---

## Evaluation

### BEIR

```bash
# Encode corpus and queries
sbatch eval.beir/encode.modernbert.sh

# Search (produces .run files)
# then compute nDCG@10 per dataset
```

### CRUX-MDS (multi-document summarization)

```bash
sbatch eval.crux/encode.modernbert.sh  # (or .bert.sh / .repllama.sh)
# search + evaluate with crux.evaluation.rac_eval
sbatch eval.crux/search.modernbert.sh
```

Metrics reported: `nDCG@10`, `alpha_nDCG@10`, `Cov@10`, `P@10`.

### MS-MARCO passage

```bash
sbatch eval.msmarco-passage/encode-d.bert.sh
sbatch eval.msmarco-passage/encode-q.bert.sh
sbatch eval.msmarco-passage/search.bert.sh
```

### NeuCLIR

```bash
sbatch eval.neuclir/encode.modernbert.sh
sbatch eval.neuclir/search.modernbert.sh
```

Baseline results (BM25, LSR, Qwen3 ± LLM reranking) are logged in `eval.neuclir/baseline.md`.

---

## Models & datasets (HuggingFace)

| Artifact | HF identifier |
|---|---|
| Base model (unsupervised ModernBERT) | `nomic-ai/modernbert-embed-base-unsupervised` |
| MS-MARCO PFT checkpoint | `DylanJHJ/nomic.modernbert-base.msmarco-passage.10k` |
| CRUX-Researchy flat checkpoint | `DylanJHJ/nomic.modernbert-base.crux-researchy-flatten.10k` |
| Training data | `DylanJHJ/crux-researchy`, `DylanJHJ/crux-researchy-new` |
| Training corpus | `DylanJHJ/crux-researchy-corpus` |
| KD training data | `DylanJHJ/crux-researchy-kdnew-ext` |
| BEIR corpus | `DylanJHJ/beir-corpus` |
| BEIR queries | `DylanJHJ/beir-subset` |

---

## Requirements

- [Tevatron](https://github.com/texttron/tevatron) (included as `tevatron/` submodule)
- [CRUX](https://github.com/DylanJHJ/crux) evaluation toolkit (`pip install crux` or local install)
- PyTorch + HuggingFace Transformers / Accelerate
- DeepSpeed (optional, configs in `configs/`)

---

## Citation

```bibtex
@article{cover2025,
  title   = {Search for Coverage: Learning Coverage-Aware Retrieval with Augmented Sub-Question Answerability},
  author  = {},
  year    = {2025}
}
```
