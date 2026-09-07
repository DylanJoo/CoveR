#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/encode.out
#SBATCH --error=logs/encode.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=/home/dju/datasets/crux

# model_dir=DylanJHJ/dpr.modernbert-base.msmarco-passage.25k
model_dir=/home/dju/models/ablation.transfer-learning/modernbert-two-stage.pos_20.neg_51.filtered.b64_n512.1e-4.512.self-prefinetuned
output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
mkdir -p $output_dir

for subset in crux-mds-duc04 crux-mds-multi_news;do
    topic_path=/home/dju/SCOPE/src/mqr/${subset}.subquestions.jsonl
    python -m tevatron.retriever.driver.encode \
      --output_dir=temp \
      --tokenizer_name answerdotai/ModernBERT-base \
      --model_name_or_path $model_dir \
      --pooling mean --normalize --bf16 \
      --per_device_eval_batch_size 256 \
      --dataset_path $topic_path \
      --encode_output_path $output_dir/query_emb.${subset}.mqr.pkl \
      --query_max_len 128 \
      --encode_is_query
done
