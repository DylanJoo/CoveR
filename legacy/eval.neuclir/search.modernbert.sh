#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=${HOME}/datasets/crux
MODEL_DIRS=(
# nomic-ai/modernbert-embed-base
facebook/contriever-msmarco
)

for model_dir in "${MODEL_DIRS[@]}"; do
    output_dir=${HOME}/indices/neuclir1/${model_dir##*/}
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

