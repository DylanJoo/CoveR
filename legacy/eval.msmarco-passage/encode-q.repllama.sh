#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=m-enc-query.out
#SBATCH --error=m-enc-query.err
#SBATCH --partition=dev-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=0-02:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

model_dir=${HOME}/models/repllama-msmarco-psg.b128_n512.1e-4
checkpoint=checkpoint-4000
output_dir=${HOME}/indices/msmarco-passage-corpus-new/${model_dir##*/}
mkdir -p $output_dir

for split in dl19 dl20;do
    export CUDA_VISIBLE_DEVICES=0
    export HIP_VISIBLE_DEVICES=0
    singularity exec $SIF  \
    python -m tevatron.retriever.driver.encode  \
      --output_dir=temp \
      --tokenizer_name meta-llama/Llama-3.1-8B-Instruct \
      --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
      --lora_name_or_path $model_dir/$checkpoint \
      --lora \
      --bf16 \
      --per_device_eval_batch_size 128 \
      --normalize \
      --pooling last  \
      --query_prefix "query: " \
      --append_eos_token \
      --query_max_len 32 \
      --dataset_name Tevatron/msmarco-passage-new \
      --corpus_name Tevatron/msmarco-passage-new \
      --dataset_split $split \
      --encode_output_path $output_dir/query_emb.$split..pkl \
      --encode_is_query
done
wait
