#!/bin/bash -l
#SBATCH --job-name=ctr
#SBATCH --output=logs/ctr.out.%a
#SBATCH --error=logs/ctr.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0,1,4,6-12%2
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

model_dir=naver/splade-v3
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

echo Encoding $DATASET corpus
python -m tevatron.retriever.driver.encode_lsr \
    --model_name_or_path $model_dir \
    --dataset_name DylanJHJ/beir-corpus \
    --dataset_split $DATASET \
    --collection_output ${HOME}/datasets/beir-corpus/indices/splade-v3/vectors.jsonl \
    --per_device_eval_batch_size 256 \
    --passage_max_len 512 \
    --quantization_factor 100
