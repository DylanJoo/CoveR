#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=enc-doc.out.%j
#SBATCH --error=enc-doc.err.%j
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-7%2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate pyserini

model_dir=DylanJHJ/dpr.bert-base-uncased.msmarco-passage.25k
output_dir=${HOME}/indices/${model_dir##*/}

mkdir -p $output_dir

# Start experimentss
SHARD_ID=$SLURM_ARRAY_TASK_ID
python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name bert-base-uncased \
    --model_name_or_path $model_dir \
    --per_device_eval_batch_size 128 \
    --passage_max_len 256 \
    --bf16 \
    --exclude_title \
    --dataset_name Tevatron/msmarco-passage-corpus-new \
    --corpus_name Tevatron/msmarco-passage-corpus-new \
    --encode_output_path $output_dir/corpus_emb.${SHARD_ID}.pkl \
    --dataset_number_of_shards 8 \
    --attn_implementation sdpa \
    --dataset_shard_index ${SHARD_ID}
