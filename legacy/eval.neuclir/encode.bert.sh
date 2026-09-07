#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/encode.out.%a
#SBATCH --error=logs/encode.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --array=0-2%3
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

model_dir=DylanJHJ/dpr.bert-base-uncased.msmarco-passage.25k
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
        --tokenizer_name bert-base-uncased \
        --model_name_or_path $model_dir \
        --per_device_eval_batch_size 1600 \
        --pooling cls --bf16 \
        --passage_max_len 512 \
        --dataset_path ${HOME}/datasets/neuclir1/${LANG}.processed_output.jsonl.gz \
        --encode_output_path $output_dir/corpus_emb.${LANG}-${shard_idx}.pkl \
        --dataset_number_of_shards 2 \
        --dataset_shard_index $shard_idx \
        --attn_implementation sdpa
done

topic_path=/home/dju/datasets/crux/crux-neuclir/topic/neuclir24-test-request.jsonl
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name bert-base-uncased \
        --model_name_or_path $model_dir \
        --pooling cls --bf16 \
        --per_device_eval_batch_size 64 \
        --dataset_path $topic_path \
        --attn_implementation sdpa \
        --encode_output_path $output_dir/query_emb.pkl \
        --query_max_len 128 \
        --encode_is_query
fi
