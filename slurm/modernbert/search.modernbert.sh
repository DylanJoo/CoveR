#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --partition=debug           # partition name
#SBATCH --ntasks-per-node=1       # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                 # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=0-00:30:00         # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

model_dir=${HOME}/models/modernbert-msmarco-psg.b64_n512.1e-4
checkpoint=checkpoint-25000
output_dir=${HOME}/indices/${model_dir##*/}

echo model: $model_dir, checkpoint: $checkpoint
for split in dl19 dl20;do
    singularity exec $SIF \
    python -m tevatron.retriever.driver.search \
	--query_reps $output_dir/query_emb.$split.pkl \
	--passage_reps $output_dir/'corpus_emb.*.pkl' \
	--depth 100 \
	--batch_size -1 \
	--save_text \
	--save_ranking_to $output_dir/$split.run

    singularity exec $SIF \
    python -m tevatron.utils.format.convert_result_to_trec \
        --input $output_dir/$split.run \
        --output $output_dir/$split.trec
done

singularity exec $SIF python -m ir_measures msmarco-passage/trec-dl-2019 $output_dir/dl19.trec nDCG@10
singularity exec $SIF python -m ir_measures msmarco-passage/trec-dl-2020 $output_dir/dl20.trec nDCG@10
