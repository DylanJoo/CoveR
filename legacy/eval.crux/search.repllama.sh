#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=search.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00         # Run time (d-hh:mm:ss)

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate ir

crux_root=${HOME}/datasets/crux
model_dir=DylanJHJ/repllama-3.1-8b.msmarco-passage.4k
output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
mkdir -p $output_dir

for subset in crux-mds-duc04 crux-mds-multi_news;do
    echo model: $model_dir $subset
    python -m tevatron.retriever.driver.search \
	--query_reps $output_dir/query_emb.$subset.pkl \
	--passage_reps $output_dir/'corpus_emb.*.pkl' \
	--depth 100 \
	--batch_size -1 \
	--save_text \
	--save_ranking_to $output_dir/$subset.run

    python -m tevatron.utils.format.convert_result_to_trec \
    --input $output_dir/$subset.run \
    --output $output_dir/$subset.trec
done

for subset in crux-mds-duc04 crux-mds-multi_news;do
    python -m crux.evaluation.rac_eval \
        --run $output_dir/$subset.trec \
        --qrel $crux_root/$subset/qrels/div_qrels-tau3.txt \
        --filter_by_oracle \
        --judge $crux_root/$subset/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl
done
