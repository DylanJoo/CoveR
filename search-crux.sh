#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result/crux.out
#SBATCH --error=result/crux.err
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
DylanJHJ/modernbert-base.cover-5k
DylanJHJ/modernbert-base.scope-flt-cover-5k
DylanJHJ/modernbert-base.scope-flt-cover-10k
DylanJHJ/modernbert-base.relevance-10k
DylanJHJ/modernbert-base.scope-5k
)

for model_dir in "${MODEL_DIRS[@]}"; do
    echo model: $model_dir $subset
    for subset in crux-mds-duc04 crux-mds-multi_news;do
        output_dir=${HOME}/scratch/crux-mds-corpus/${model_dir##*/}
        python -m tevatron.retriever.driver.search \
            --query_reps $output_dir/query_emb.$subset.pkl \
            --passage_reps $output_dir/'corpus_emb*pkl' \
            --depth 100 \
            --batch_size -1 \
            --save_text \
            --save_ranking_to $output_dir/$subset.run

        python -m tevatron.utils.format.convert_result_to_trec \
            --input $output_dir/$subset.run \
            --output $output_dir/$subset.trec 

        python -m crux.evaluation.rac_eval \
            --run $output_dir/$subset.trec \
            --qrel $CRUX_ROOT/$subset/qrels/div_qrels-tau3.txt \
            --filter_by_oracle \
            --judge $CRUX_ROOT/$subset/judge
    done
done
