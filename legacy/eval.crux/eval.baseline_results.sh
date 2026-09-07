#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=${HOME}/datasets/crux

for baseline in ${HOME}/crux/runs/*mds-duc04.trec;do
    echo $baseline
    python -m crux.evaluation.rac_eval \
        --run $baseline \
        --qrel $CRUX_ROOT/crux-mds-duc04/qrels/div_qrels-tau3.txt \
        --filter_by_oracle \
        --judge $CRUX_ROOT/crux-mds-duc04/judge 
done

for baseline in ${HOME}/crux/runs/*mds-multi_news.trec;do
    echo $baseline
    python -m crux.evaluation.rac_eval \
        --run $baseline \
        --qrel $CRUX_ROOT/crux-mds-multi_news/qrels/div_qrels-tau3.txt \
        --filter_by_oracle \
        --judge $CRUX_ROOT/crux-mds-multi_news/judge 
done
