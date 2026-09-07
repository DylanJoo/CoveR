#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/llama.out.%a
#SBATCH --error=logs/llama.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --array=0-2%1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=/home/dju/datasets/crux

model_dir=DylanJHJ/repllama-3.1-8b.msmarco-passage.4k
output_dir=${HOME}/indices/neuclir1/${model_dir##*/}
mkdir -p $output_dir

LANGS=(
"rus"
"fas"
"zho"
)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

for shard_idx in 0 1 2 3;do
    echo Encoding $LANG $shard_idx
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --lora_name_or_path $model_dir \
        --lora \
        --bf16 \
        --per_device_eval_batch_size 16 \
        --normalize \
        --pooling last \
        --passage_prefix "passage: " \
        --append_eos_token \
        --passage_max_len 2048 \
        --dataset_path ${HOME}/datasets/neuclir1/${LANG}.processed_output.jsonl.gz \
        --encode_output_path $output_dir/corpus_emb.${LANG}-${shard_idx}.pkl \
        --dataset_number_of_shards 4 \
        --dataset_shard_index $shard_idx
done

topic_path=/home/dju/datasets/crux/crux-neuclir/topic/neuclir24-test-request.jsonl
if [[ $LANG -eq "rus" ]]; then
    echo "Encoding $topic_path"
    python -m tevatron.retriever.driver.encode \
        --tokenizer_name meta-llama/Llama-3.1-8B-Instruct \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --lora_name_or_path $model_dir \
        --lora \
        --bf16 \
        --per_device_eval_batch_size 8 \
        --normalize \
        --pooling last  \
        --query_prefix "query: " \
        --append_eos_token \
        --dataset_path $topic_path \
        --encode_output_path $output_dir/query_emb.pkl \
        --query_max_len 256 \
        --encode_is_query
fi
