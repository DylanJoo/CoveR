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

model_dir=DylanJHJ/dpr.modernbert-base.msmarco-passage.25k
# model_dir=DylanJHJ/dpr.modernbert-base.crux-researchy-flatten.25k
# model_dir=/home/dju/models/ablation.cov-sampling/modernbert-crux-researchy-pos_20.neg_51.filtered.b64_n512.1e-4.512
# model_dir=/home/dju/models/ablation.cov-sampling/modernbert-crux-researchy-pos_half.neg_zero.b64_n512.1e-4.512
# model_dir=/home/dju/models/ablation.cov-sampling/modernbert-crux-researchy-pos_high.neg_zero.b64_n512.1e-4.512
# model_dir=/home/dju/models/ablation.cov-sampling/modernbert-crux-researchy-pos_low.neg_zero.b64_n512.1e-4.512
# model_dir=/home/dju/models/ablation.cov-sampling/modernbert-crux-researchy-pos_high.neg_low.b64_n512.1e-4.512
# model_dir=/home/dju/models/ablation.cov-sampling/modernbert-crux-researchy-pos_high.neg_quarter.b64_n512.1e-4.512
# model_dir=/home/dju/models/ablation.cov-sampling/modernbert-crux-researchy-pos_zero.neg_high.b64_n512.1e-4.512

# model_dir=/home/dju/models/ablation.transfer-learning/modernbert-two-stage.pos_20.neg_51.filtered.b64_n512.1e-4.512.self-prefinetuned
# model_dir=/home/dju/models/ablation.transfer-learning/modernbert-two-stage.pos_20.neg_51.filtered.b64_n512.1e-4.512

# model_dir=/home/dju/models/ablation.transfer-learning/modernbert-mixed-dataset.crux-researchy-pos_high.neg_zero.b64_n512.1e-4.512
# model_dir=/home/dju/models/ablation.transfer-learning/modernbert-mixed-dataset.crux-researchy-pos_high.neg_zero.b64_n512.1e-4.512.35k

output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
mkdir -p $output_dir

for lambda in 1.0 0.99 0.98 0.97 0.96 0.95 0.90 0.89 0.88 0.87 0.86 0.85 0.84 0.83 0.82 0.81 0.80;do

    for subset in crux-mds-duc04 crux-mds-multi_news;do
        echo model: $model_dir $subset
        python -m tevatron.retriever.driver.search.diversify.mmr \
            --query_reps $output_dir/query_emb.$subset.pkl \
            --passage_reps $output_dir/'corpus_emb.*.pkl' \
            --depth 100 \
            --batch_size -1 \
            --save_text \
            --save_ranking_to $output_dir/$subset.run \
            --lambda_param $lambda

        python -m tevatron.utils.format.convert_result_to_trec \
            --input $output_dir/$subset.run \
            --output $output_dir/$subset.mmr-$lambda.trec
    done

    for subset in crux-mds-duc04 crux-mds-multi_news;do
        python -m crux.evaluation.rac_eval \
            --run $output_dir/$subset.mmr-$lambda.trec \
            --qrel $CRUX_ROOT/$subset/qrels/div_qrels-tau3.txt \
            --filter_by_oracle \
            --judge $CRUX_ROOT/$subset/judge 
    done
done
