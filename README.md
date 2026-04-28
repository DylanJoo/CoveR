# CoveR

**CoveR** is the codebase for _"Search for Coverage: Learning Coverage-Aware Retrieval with Augmented Sub-Question Answerability"_.

The goal is to train dense retrieval models that maximise **information coverage**, the fraction of answerable sub-topics in a multi-aspect information need that the retrieved documents collectively address — rather than simply optimising for relevance-based ranking.

Standard dense retrievers are trained on relevance-annotated pairs (e.g., MS-MARCO). For complex, research-style requests that span multiple sub-aspects, a document may be relevant to the request as a whole but only partially address its sub-topics. CoveR trains on coverage-labelled data derived from [CRUX-Researchy](https://huggingface.co/datasets/DylanJHJ/crux-researchy), where each document is judged on how many of a request's sub-questions it can answer.

---

## Installation

```bash
# Install Tevatron
pip install transformers datasets peft
pip install deepspeed accelerate
pip install faiss-cpu
cd tevatron && pip install -e . && cd ..

# Install the CRUX evaluation toolkit
pip install crux   # or: git clone https://github.com/DylanJHJ/crux && pip install -e crux/
```

## Overview

| | |
|---|---|
| [Models & datasets](#models--datasets-huggingface)                                              | HuggingFace checkpoints and training corpora |
| [Evaluation](#evaluation)                                                                        | BEIR · CRUX-MDS · MS-MARCO passage · NeuCLIR |
| [Pre-fine-tuning (PFT)](#1-pre-fine-tuning-pft--optional)                                       | Contrastive training on MS-MARCO (relevance-based) |
| [Coverage-based training (CoveR)](#2-coverage-based-training-cover)                             | Coverage-bucket sampling pairs from CRUX-Researchy |
| [Sub-question augmented training (CoveR + SQ)](#3-sub-question-augmented-training-cover--sq)    | CoveR + sub-question augmentation (w/ or w/o PFT) |
| [Two-stage training](#4-two-stage-training)                                                     | Flat relevance CRUX fine-tuning → coverage fine-tuning |
| [Data curation](#data-curation)                                                                  | Coverage-bucket sampling · sub-question generation · MS-MARCO sampling |

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
## Training pipeline

### 1. Pre-fine-tuning (PFT) — optional

Fine-tune a base model on MS-MARCO passage retrieval as a warm-start before coverage training.

```bash
accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 --num_processes 2 \
    tevatron.retriever.driver.train_dev \
    --model_name_or_path nomic-ai/modernbert-embed-base-unsupervised \
    --output_dir <output_dir> \
    --dataset_name Tevatron/msmarco-passage-new \
    --corpus_name Tevatron/msmarco-passage-corpus-new \
    --per_device_train_batch_size 32 --train_group_size 8 \
    --bf16 --pooling mean --normalize \
    --passage_prefix "search_document: " --query_prefix "search_query: " \
    --temperature 0.02 --learning_rate 1e-4 \
    --query_max_len 32 --passage_max_len 256 \
    --max_steps 10000 --warmup_steps 1000 \
    --lr_scheduler_type cosine --weight_decay 0.01 \
    --exclude_title
```

The SLURM scripts `slurm/modernbert/train.modernbert.sh` (relevance-based) and `slurm/modernbert/train.modernbert.kd.sh` (with KD scores from Qwen3-0.6B reranker) run the equivalent commands on the cluster.

The resulting checkpoint (`DylanJHJ/nomic.modernbert-base.msmarco-passage.10k` or similar) is used as the starting point for coverage training.

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

Uses the full research request **and** decomposed sub-queries as dual query views (`--subquery_prefix`). CovDistil KLD loss can be enabled via `--covdistil_lambda`.

**SCOPE w/o PFT** — train directly from the unsupervised base:

```bash
accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 --num_processes 4 \
    tevatron.retriever.driver.train_dualdistil \
    --model_name_or_path nomic-ai/modernbert-embed-base-unsupervised \
    --output_dir <output_dir> \
    --dataset_name DylanJHJ/crux-researchy-kdnew-ext \
    --corpus_name DylanJHJ/crux-researchy-corpus \
    --dataset_split pos_half.neu_low.neg_zero \
    --per_device_train_batch_size 16 --train_group_size 8 \
    --bf16 --pooling mean --normalize \
    --passage_prefix "search_document: " \
    --query_prefix "search_query: " --subquery_prefix "search_query: " \
    --temperature 0.02 --learning_rate 1e-4 \
    --use_crossentropy 1.0 --contrastive_lambda 1.0 --sq_contrastive_lambda 1.0 \
    --covdistil_method KLD --covdistil_lambda 0.0 \
    --query_max_len 180 --passage_max_len 512 \
    --max_steps 10000 --warmup_steps 500 \
    --lr_scheduler_type cosine --exclude_title
```

**SCOPE w/ PFT** — warm-start from the MS-MARCO checkpoint:

```bash
# Same command as above, but replace --model_name_or_path with the PFT checkpoint:
#   --model_name_or_path DylanJHJ/nomic.modernbert-base.msmarco-passage.10k
```

The SLURM scripts `slurm/exp-for-init/unsupervised-pft.scope-ft.sh` and `slurm/exp-for-init/msmarco-pft.scope-ft.sh` run the equivalent commands on the cluster.

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

## Citation

```bibtex
@article{cover2025,
  title   = {Search for Coverage: Learning Coverage-Aware Retrieval with Augmented Sub-Question Answerability},
  author  = {},
  year    = {2025}
}
```
