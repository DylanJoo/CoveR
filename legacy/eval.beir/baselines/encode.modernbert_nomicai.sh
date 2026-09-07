#!/bin/bash -l
#SBATCH --job-name=nomicai
#SBATCH --output=logs/nomicai.out.%a
#SBATCH --error=logs/nomicai.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-12%2
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

model_dir=nomic-ai/modernbert-embed-base-unsupervised
corpus_name=beir-corpus
output_dir=${HOME}/indices/${corpus_name}/${model_dir##*/}
mkdir -p $output_dir

DATASETS=(
"beir.arguana"
"beir.climate_fever"
"beir.dbpedia_entity"
"beir.fever"
"beir.fiqa"
"beir.hotpotqa"
"beir.nfcorpus"
"beir.nq"
"beir.quora"
"beir.scidocs"
"beir.scifact"
"beir.trec_covid"
"beir.webis_touche2020"
)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

for SHARD_ID in 0 1;do
    echo Encoding $DATASET corpus $SHARD_ID
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name answerdotai/ModernBERT-base \
        --model_name_or_path $model_dir \
        --per_device_eval_batch_size 3840 \
        --pooling mean --bf16 --normalize \
        --passage_max_len 512 \
        --passage_prefix "search_document: " \
        --dataset_name DylanJHJ/${corpus_name} \
        --dataset_split $DATASET \
        --encode_output_path $output_dir/corpus_emb.${DATASET}-${SHARD_ID}.pkl \
        --dataset_shard_index ${SHARD_ID} \
        --dataset_number_of_shards 2
done

echo Encoding $DATASET queries
python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name answerdotai/ModernBERT-base \
    --model_name_or_path $model_dir \
    --per_device_eval_batch_size 128 \
    --pooling mean --bf16 --normalize \
    --query_prefix "search_query: " \
    --dataset_name  DylanJHJ/beir-subset \
    --dataset_split $DATASET \
    --encode_output_path $output_dir/query_emb.${DATASET}.pkl \
    --query_max_len 128 \
    --encode_is_query
