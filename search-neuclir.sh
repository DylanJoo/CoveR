#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result/neuclir.out
#SBATCH --error=result/neuclir.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00

# ENV
source ${HOME}/.bashrc
initconda
conda activate inference 

CRUX_ROOT=${HOME}/datasets/crux
MODEL_DIRS=(
# DylanJHJ/modernbert-base.cover-5k
# DylanJHJ/modernbert-base.relevance-10k
# DylanJHJ/modernbert-base.scope-5k
# DylanJHJ/modernbert-base.scope-flt-cover-5k
# DylanJHJ/modernbert-base.cover-5k.reproduced
DylanJHJ/modernbert-base.scope-flt-cover-10k # R
)

for model_dir in "${MODEL_DIRS[@]}"; do
    output_dir=${HOME}/scratch/neuclir1/${model_dir##*/}
    mkdir -p $output_dir
    python -m tevatron.retriever.driver.search \
        --query_reps $output_dir/query_emb.pkl \
        --passage_reps $output_dir/'corpus_emb*pkl' \
        --depth 100 \
        --batch_size -1 \
        --save_text \
        --save_ranking_to $output_dir/neuclir24-test.run

    python -m tevatron.utils.format.convert_result_to_trec \
        --input $output_dir/neuclir24-test.run \
        --output $output_dir/neuclir24-test.trec

    echo $output_dir/neuclir24-test.trec
    python -m crux.evaluation.rac_eval \
        --run $output_dir/neuclir24-test.trec \
        --qrel $CRUX_ROOT/crux-neuclir/qrels/neuclir24-test-request.qrel \
        --judge $CRUX_ROOT/crux-neuclir/judge/ratings.human.jsonl
done
