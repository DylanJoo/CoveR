#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=search.o
#SBATCH --error=search.e
#SBATCH --partition=small           # partition name
#SBATCH --ntasks-per-node=1       # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                 # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=0-00:30:00         # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate pyserini

model_dir=${HOME}/models/llama-msmarco-psg.b8
checkpoint=checkpoint-3836
output_dir=${HOME}/indices/msmarco-passage-corpus-new/${model_dir##*/}

for split in dl19 dl20;do
    python -m tevatron.retriever.driver.search \
	--query_reps $output_dir/query_emb.$split..pkl \
	--passage_reps $output_dir/'corpus_emb.*.pkl' \
	--depth 100 \
	--batch_size -1 \
	--save_text \
	--save_ranking_to $output_dir/$split.run

    python -m tevatron.utils.format.convert_result_to_trec \
        --input $output_dir/$split.run \
        --output $output_dir/$split.trec
done

python -m ir_measures msmarco-passage/trec-dl-2019 $output_dir/dl19.trec nDCG@10
python -m ir_measures msmarco-passage/trec-dl-2020 $output_dir/dl20.trec nDCG@10
