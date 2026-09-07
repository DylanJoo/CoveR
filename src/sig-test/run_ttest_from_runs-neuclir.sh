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

model_dir=${HOME}/models/main.learning/ce_1.0-selfdistil_0.1
model_dir=${HOME}/models/main.learning/modernbert-two-stage-crux-researchy-pos_half.neg_zero.b64_n512.1e-4.crux-researchy
output_dira=${HOME}/indices/neuclir1/${model_dir##*/}

model_dir=nomic-ai/modernbert-embed-base-unsupervised
model_dir=DylanJHJ/nomic.modernbert-base.msmarco-passage.10k
output_dirb=${HOME}/indices/neuclir1/${model_dir##*/}

python run_ttest_from_runs-neuclir.py \
    --qrel $CRUX_ROOT/crux-neuclir/qrels/neuclir24-test-request.qrel \
    --judge $CRUX_ROOT/crux-neuclir/judge/ratings.human.jsonl \
    --run_a $output_dira/neuclir24-test.trec --run_b $output_dirb/neuclir24-test.trec

