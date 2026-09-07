#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result.%a
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-12%13
#SBATCH --mem=48G
#SBATCH --time=00:30:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

model_dir=DylanJHJ/dpr.bert-base-uncased.msmarco-passage.25k
output_dir=${HOME}/indices/beir-corpus/${model_dir##*/}
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

QRELS=(
"beir/arguana"
"beir/climate-fever"
"beir/dbpedia-entity/test"
"beir/fever/test"
"beir/fiqa/test"
"beir/hotpotqa/test"
"beir/nfcorpus/test"
"beir/nq"
"beir/quora/test"
"beir/scidocs"
"beir/scifact/test"
"beir/trec-covid"
"beir/webis-touche2020/v2"
)
irds_tag=${QRELS[$SLURM_ARRAY_TASK_ID]}

python -m tevatron.retriever.driver.search \
    --query_reps $output_dir/query_emb.${DATASET}.pkl \
    --passage_reps $output_dir/corpus_emb.${DATASET}.pkl \
    --depth 100 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to $output_dir/${DATASET}.run

python -m tevatron.utils.format.convert_result_to_trec \
    --input $output_dir/${DATASET}.run \
    --output $output_dir/${DATASET}.trec

result=$(python -m ir_measures $irds_tag $output_dir/${DATASET}.trec nDCG@10)

short_name=$(basename "$DATASET" | cut -c6-8)
echo "${short_name} | $result"
