# CoveR

The codebase for _"Search for Coverage: Learning Coverage-Aware Retrieval with Augmented Sub-Question Answerability"_.

Standard dense retrievers are trained on relevance-annotated pairs (e.g., MS-MARCO). For complex, research-style requests that span multiple sub-aspects, a document may be broadly relevant but only partially address the request's sub-topics. CoveR instead trains on coverage-labelled data from [SCOPE](https://huggingface.co/datasets/DylanJHJ/SCOPE), where each document is sampled with how many of a request's sub-questions it can answer.

---

## Overview

| | |
|---|---|
| [Installation](#installation)                                                                 | Setup and package dependencies |
| [Usage](#usage)                                                                               | Encoding queries and documents with the model |
| [Models & datasets](#models--datasets-huggingface)                                            | HuggingFace checkpoints and training corpora |
| [Evaluation](#evaluation)                                                                     | BEIR · CRUX-MDS · NeuCLIR |
| [Pre-fine-tuning (PFT)](#pre-fine-tuning-pft)                                                 | Contrastive training on MS-MARCO (relevance-based) |
| [Coverage-based training (CoveR)](#coverage-based-training-cover)                             | Coverage-bucket sampling pairs from SCOPE |
| [Data curation](#data-curation)                                                               | Coverage-bucket sampling · sub-question generation · MS-MARCO sampling |

## Installation

```bash
# Install Tevatron
pip install transformers datasets peft
pip install deepspeed accelerate
pip install faiss-cpu ir_datasets ir_measures
cd tevatron && pip install -e . && cd ..

# Install the CRUX evaluation toolkit and download the datasets
git clone https://github.com/DylanJHJ/crux && pip install -e crux/

cd ${HOME}/datasets/
git lfs install
git clone https://huggingface.co/datasets/DylanJHJ/crux
git clone https://huggingface.co/datasets/DylanJHJ/crux-mds-corpus
```

> For BEIR corpus, we use `ir_datasets` API at [here](https://ir-datasets.com/beir.html)
> For NeuCLIR corpus, we use the [English translation provided by the organizer team](https://huggingface.co/datasets/neuclir/neuclir1/viewer/mt_docs).

---
## Usage

The CoveR models use **mean pooling** over the last hidden states with **L2 normalisation**, matching how Tevatron encodes queries and passages. Queries should be prefixed with `"search_query: "` and documents with `"search_document: "`.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "DylanJHJ/modernbert-base.cover-5k"
QUERY_PREFIX = "search_query: "
DOC_PREFIX   = "search_document: "

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).eval()

def mean_pool_normalize(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).bool()
    masked = hidden_states.masked_fill(~mask, 0.0)
    reps = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    return F.normalize(reps, p=2, dim=-1)


def encode(texts: list[str], prefix: str, max_length: int = 512) -> torch.Tensor:
    texts = [prefix + t for t in texts]
    batch = tokenizer(texts, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        output = model(**batch, return_dict=True)
    return mean_pool_normalize(output.last_hidden_state, batch["attention_mask"])

query = [
    "I need a report on suicide rates in Japan during the COVID-19 pandemic. "
    "This would include cases in which the pandemic is identified as a causal factor, "
    "but also other factors from that period even if not clearly associated with the pandemic. "
    "I am not looking for information on suicide rates in Japan more generally, however, "
    "unless that is used as a basis for comparison with rates during the pandemic."
]
docs  = [
    "Japan saw a notable rise in suicide rates in 2020, ..."
    "a trend attributed to pandemic-related social isolation, job losses ...",
    "Government data show that suicide deaths in Japan increased by approximately 4% ..."
]

q_emb = encode(query, QUERY_PREFIX, max_length=180)   # (1, H)
d_emb = encode(docs,  DOC_PREFIX,  max_length=512)    # (3, H)
scores = (q_emb @ d_emb.T).squeeze(0)                 # (3,)
```

---
## Models & datasets (HuggingFace)

| Artifact | HF identifier |
|---|---|
| Unsupervised            | `nomic-ai/modernbert-embed-base-unsupervised` |
| Relevance               | `DylanJHJ/modernbert-base.relevance-10k` |
| CoveR (with pFT)        | `DylanJHJ/modernbert-base.cover-5k` |
| Training data           | `DylanJHJ/crux-researchy`, `DylanJHJ/crux-researchy-new` |
| Training corpus         | `DylanJHJ/crux-researchy-corpus` |
| BEIR corpus             | `DylanJHJ/beir-corpus` |
| BEIR queries            | `DylanJHJ/beir` |
| NeuCLIR corpus (mt)     | `https://huggingface.co/datasets/neuclir/neuclir1/viewer/mt_docs` |
| NeuCLIR queries         | |
| CRUX corpus             | `DylanJHJ/crux-mds-corpus` |
| CRUX queries            | `DylanJHJ/crux` |

---
## Evaluation

See the following scripts for details: `encode-beir.sh` and `search-beir.sh`; `encode-neuclir.sh` and `search-neuclir.sh`

### BEIR

Encode the corpus (two shards per dataset) and queries, then retrieve and evaluate with `ir_measures`.

```bash
MODEL=DylanJHJ/modernbert-base.cover-5k
OUTPUT_DIR=${HOME}/scratch/beir-corpus/${MODEL##*/}
DATASET=beir.nfcorpus   # replace with target dataset

# Encode corpus (2 shards)
for SHARD_ID in 0 1; do
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name answerdotai/ModernBERT-base \
        --model_name_or_path $MODEL \
        --per_device_eval_batch_size 2048 \
        --passage_max_len 512 \
        --pooling mean --normalize --bf16 \
        --passage_prefix "search_document: " \
        --dataset_name DylanJHJ/beir-corpus \
        --dataset_split $DATASET \
        --encode_output_path $OUTPUT_DIR/corpus_emb.${DATASET}-${SHARD_ID}.pkl \
        --dataset_shard_index ${SHARD_ID} \
        --dataset_number_of_shards 2
done

# Encode queries
python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name answerdotai/ModernBERT-base \
    --model_name_or_path $MODEL \
    --pooling mean --normalize --bf16 \
    --query_prefix "search_query: " \
    --per_device_eval_batch_size 128 \
    --dataset_name DylanJHJ/beir-subset \
    --dataset_split $DATASET \
    --encode_output_path $OUTPUT_DIR/query_emb.${DATASET}.pkl \
    --query_max_len 256 \
    --encode_is_query

# Retrieve and evaluate
python -m tevatron.retriever.driver.search \
    --query_reps $OUTPUT_DIR/query_emb.${DATASET}.pkl \
    --passage_reps "$OUTPUT_DIR/corpus_emb.${DATASET}*pkl" \
    --depth 100 --batch_size -1 --save_text \
    --save_ranking_to $OUTPUT_DIR/${DATASET}.run

python -m tevatron.utils.format.convert_result_to_trec \
    --input $OUTPUT_DIR/${DATASET}.run \
    --output $OUTPUT_DIR/${DATASET}.trec

python -m ir_measures beir/<dataset-name> $OUTPUT_DIR/${DATASET}.trec nDCG@10
```

### NeuCLIR
Follow CRUX, we use the provided evaluation script to compute alpha ndcg and coverage at 10.
```bash
CRUX_ROOT=${HOME}/datasets/crux
python -m crux.evaluation.rac_eval \
    --run $OUTPUT_DIR/neuclir24-test.trec \
    --qrel $CRUX_ROOT/crux-neuclir/qrels/neuclir24-test-request.qrel \
    --judge $CRUX_ROOT/crux-neuclir/judge/ratings.human.jsonl
```

The other reproduced baselines can be found in [runs-and-qrels](http://) _(link TBD)_.

---
## Training pipeline

### Pre-fine-tuning (PFT)
Fine-tune a base model on MS-MARCO passage retrieval as a relevance baseline. This stage also serves as pre-fine-tuning (PFT) for coverage training — the resulting checkpoint is used as the starting point for subsequent coverage training.

```bash
accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 --num_processes 2 \
    tevatron.retriever.driver.train_dev \
    --model_name_or_path nomic-ai/modernbert-embed-base-unsupervised \
    --output_dir unsupervised.msmarco-passage-new.10k \
    --dataset_name Tevatron/msmarco-passage-new \
    --corpus_name Tevatron/msmarco-passage-corpus-new \
    --per_device_train_batch_size 32 \
    --train_group_size 8 \
    --bf16 --pooling mean --normalize \
    --passage_prefix "search_document: " \
    --query_prefix "search_query: " \
    --temperature 0.02 \
    --learning_rate 1e-4 \
    --query_max_len 32 \
    --passage_max_len 256 \
    --max_steps 10000 \
    --warmup_steps 1000 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --exclude_title
```
Also, the training scripts can be found: `slurm/unsupervised.msmarco.sh`; and `slurm/unsupervised.scope-flt.sh` for the SCOPE-flatten variant.
> For the SCOPE-flatten setting, change the dataset and set `passage_max_len` to 512. The total batch size remains 64 (16 per device × 4 processes).

### Coverage-based training (CoveR)
Train on SCOPE using coverage-based contrastive learning. Set `model_name_or_path` to the PFT checkpoint (for the PFT → CoveR pipeline) or a raw unsupervised base model. For example,
**CoveR (w/ pft on msmarco)** - train from our reproduced relevance checkpoints: `dylanjhj/modernbert-base.relevance-10k` as the initilization.
**SCOPE (w/o pFT)** — train directly from the unsupervised base: 

The CovDistil KLD loss can be enabled via `--covdistil_lambda` with the control hyperparameter `--covdistil_lambda`
Also, the training scripts can be found: `slurm/relevance-ms-pft.scope-5k.sh`; the script without PFT is at `slurm/unsupervised.scope-5k.sh`.

```bash
lr=1e-4
model_dir=${HOME}/models/CoveR/relevance-ms-pft.cover-5k
PRETRAINED=DylanJHJ/modernbert-base.relevance-10k

accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train_dualdistil \
    --exclude_title \
    --output_dir ${model_dir} \
    --model_name_or_path $PRETRAINED \
    --save_steps 1000 \
    --dataset_name DylanJHJ/crux-researchy-kdnew-ext \
    --corpus_name DylanJHJ/crux-researchy-corpus \
    --request_as_query True \
    --dataset_split pos_half.neu_low.neg_zero \
    --per_device_train_batch_size 16 \
    --train_group_size 8 \
    --prediction_loss_only True \
    --bf16 --pooling mean --normalize \
    --passage_prefix "search_document: " \
    --query_prefix "search_query: " \
    --subquery_prefix "search_query: " \
    --temperature 0.02 \
    --use_crossentropy 1.0 \
    --use_kld 0.0 \
    --contrastive_lambda 1.0 \
    --sq_contrastive_lambda 0.0 \
    --covdistil_method KLD \
    --covdistil_lambda 0.1 \
    --eval_steps 500 \
    --learning_rate $lr \
    --query_max_len 180 \
    --passage_max_len 512 \
    --dataloader_num_workers 8 \
    --lr_scheduler_type 'cosine' \
    --weight_decay 0.01 \
    --max_steps 5000 \
    --warmup_steps 500 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --run_name ${model_dir##*/}
```

### Data curation
The training data is split into several coverage-bucket subsets. `pos_half.neg_zero` yields the best results among them.
See the HuggingFace repository for dataset access. Note that the corpus is sourced from ClueWeb Category B, which requires a separate license.

| Split | Positives | Negatives |
|---|---|---|
| `pos_high.neg_zero` | cov ≥ 0.75 | cov = 0 |
| `pos_high.neg_low` | cov ≥ 0.75 | 0 ≤ cov < 0.25 |
| `pos_high.neg_quarter` | cov ≥ 0.75 | 0 ≤ cov < 0.5 |
| `pos_half.neg_zero` | 0.5 ≤ cov < 0.75 | cov = 0 |
| `pos_low.neg_zero` | 0 ≤ cov < 0.25 | cov = 0 |


---

## Citation

```bibtex
@article{ju2026cover,
  title   = {Search for Coverage: Learning Coverage-Aware Retrieval with Augmented Sub-Question Answerability},
  author  = {Ju, Jia-Huei and Yang, Eugene and Adriaanse, Trevor and Verberne, Suzan and Yates, Andrew},
  journal = {arXiv preprint arXiv:2605.28522},
  year    = {2026}
}
```
