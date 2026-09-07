#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=enc-query.out
#SBATCH --error=enc-query.err
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-0%1
#SBATCH --mem=64G
#SBATCH --time=0-00:10:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate pyserini

model_dir=DylanJHJ/dpr.bert-base-uncased.msmarco-passage-titled.25k
model_dir=DylanJHJ/dpr.bert-base-uncased.msmarco-passage.25k
output_dir=${HOME}/indices/${model_dir##*/}

mkdir -p $output_dir

for split in dl19 dl20 dev;do
    python -m tevatron.retriever.driver.encode \
      --output_dir=temp \
      --tokenizer_name bert-base-uncased \
      --model_name_or_path $model_dir \
      --bf16 \
      --per_device_eval_batch_size 1024 \
      --dataset_name Tevatron/msmarco-passage-new \
      --dataset_split $split \
      --attn_implementation sdpa \
      --encode_output_path $output_dir/$split.query_emb.pkl \
      --query_max_len 32 \
      --encode_is_query
done
