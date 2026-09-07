#!/bin/bash -l
#SBATCH --job-name=nomic
#SBATCH --output=logs/nomic.out.%a
#SBATCH --error=logs/nomic.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --array=0-2%1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

model_dir=nomic-ai/modernbert-embed-base
output_dir=${HOME}/indices/neuclir1/${model_dir##*/}
mkdir -p $output_dir

LANGS=(
"rus"
"fas"
"zho"
)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

for shard_idx in 0 1;do
    echo Encoding $LANG $shard_idx
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name answerdotai/ModernBERT-base \
        --model_name_or_path $model_dir \
        --per_device_eval_batch_size 1024 \
        --passage_prefix "search_document: " \
        --pooling mean --bf16 --normalize \
        --passage_max_len 1024 \
        --dataset_path ${HOME}/datasets/neuclir1/${LANG}.processed_output.jsonl.gz \
        --encode_output_path $output_dir/corpus_emb.${LANG}-${shard_idx}.pkl \
        --dataset_number_of_shards 2 \
        --dataset_shard_index $shard_idx
done

topic_path=/home/dju/datasets/crux/crux-neuclir/topic/neuclir24-test-request.jsonl
if [[ $LANG -eq "rus" ]]; then
    echo "Encoding $topic_path"
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name answerdotai/ModernBERT-base \
        --model_name_or_path $model_dir \
        --pooling mean --normalize --bf16 \
        --per_device_eval_batch_size 64 \
        --query_prefix "search_query: " \
        --dataset_path $topic_path \
        --encode_output_path $output_dir/query_emb.pkl \
        --query_max_len 128 \
        --encode_is_query
