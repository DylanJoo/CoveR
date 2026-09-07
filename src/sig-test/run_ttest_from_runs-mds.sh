#!/bin/bash -l
#SBATCH --job-name=ttest
#SBATCH --output=ttest.out
#SBATCH --error=ttest.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=${HOME}/datasets/crux

model_dir=nomic-ai/modernbert-embed-base-unsupervised
model_dir=DylanJHJ/repllama-3.1-8b.msmarco-passage.4k
output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}

subset=crux-mds-duc04
python run_ttest_from_runs-mds.py
    --run $output_dir/$subset.trec \
    --qrel $CRUX_ROOT/$subset/qrels/div_qrels-tau3.txt \
    --filter_by_oracle \
    --judge $CRUX_ROOT/$subset/judge 

subset=crux-mds-multi_news
python run_ttest_from_runs-mds.py
    --run $output_dir/$subset.trec \
    --qrel $CRUX_ROOT/$subset/qrels/div_qrels-tau3.txt \
    --filter_by_oracle \
    --judge $CRUX_ROOT/$subset/judge 
