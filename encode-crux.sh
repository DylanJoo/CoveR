#!/bin/bash -l
#SBATCH --job-name=crux-encode
#SBATCH --output=logs/crux-encode.out.%a
#SBATCH --error=logs/crux-encode.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# ENV
source ${HOME}/.bashrc
initconda
conda activate inference 

MODEL_DIRS=(
DylanJHJ/modernbert-base.cover-5k # R
DylanJHJ/modernbert-base.relevance-10k # R
)

for model_dir in "${MODEL_DIRS[@]}"; do
    output_dir=${HOME}/scratch/crux-mds-corpus/${model_dir##*/}
    mkdir -p $output_dir

    for split in train test;do
        echo Encoding CRUX corpus $SHARD_ID
        python -m tevatron.retriever.driver.encode \
            --output_dir=temp \
            --tokenizer_name answerdotai/ModernBERT-base \
            --model_name_or_path $model_dir \
            --per_device_eval_batch_size 2048 \
            --passage_max_len 512 \
            --pooling mean --bf16 --normalize  \
            --passage_prefix "search_document: " \
            --exclude_title \
            --dataset_name DylanJHJ/crux-mds-corpus \
            --dataset_split $split \
            --encode_output_path $output_dir/corpus_emb.$split.pkl 
    done

    for subset in crux-mds-duc04 crux-mds-multi_news;do
    for topic_path in $CRUX_ROOT/$subset/topic/*jsonl; do # only has 1
        echo "Encoding $topic_path"
        python -m tevatron.retriever.driver.encode \
            --output_dir=temp \
            --tokenizer_name answerdotai/ModernBERT-base \
            --model_name_or_path $model_dir \
            --pooling mean --normalize --bf16 \
            --query_prefix "search_query: " \
            --per_device_eval_batch_size 128 \
            --dataset_path $topic_path \
            --encode_output_path $output_dir/query_emb.${subset}.pkl \
            --query_max_len 128 \
            --encode_is_query
    done
    done

done
