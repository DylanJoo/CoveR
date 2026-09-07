#!/bin/bash -l
#SBATCH --job-name=ablation
#SBATCH --output=logs/encode.out.%a
#SBATCH --error=logs/encode.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=7%2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh # ilps
conda activate inference 

corpus_name=beir-corpus
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

for model_dir in ${HOME}/models/ablation.two-stage/modernbert-*msmarco*;do
    output_dir=${HOME}/indices/${corpus_name}/${model_dir##*/}
    mkdir -p $output_dir

    # for SHARD_ID in 0 1;do
    #     echo Encoding $DATASET corpus $SHARD_ID
    #     python -m tevatron.retriever.driver.encode \
    #         --output_dir=temp \
    #         --tokenizer_name answerdotai/ModernBERT-base \
    #         --model_name_or_path $model_dir \
    #         --per_device_eval_batch_size 2048 \
    #         --passage_max_len 512 \
    #         --pooling mean --normalize --bf16 \
    #         --passage_prefix "search_document: " \
    #         --dataset_name DylanJHJ/${corpus_name} \
    #         --dataset_split $DATASET \
    #         --encode_output_path $output_dir/corpus_emb.${DATASET}-${SHARD_ID}.pkl \
    #         --dataset_shard_index ${SHARD_ID} \
    #         --dataset_number_of_shards 2
    # done

    echo Encoding $DATASET queries
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name answerdotai/ModernBERT-base \
        --model_name_or_path $model_dir \
        --pooling mean --normalize --bf16 \
        --query_prefix "search_query: " \
        --per_device_eval_batch_size 128 \
        --dataset_name  DylanJHJ/beir-subset \
        --dataset_split $DATASET \
        --encode_output_path $output_dir/query_emb.${DATASET}.pkl \
        --query_max_len 256 \
        --encode_is_query
done
