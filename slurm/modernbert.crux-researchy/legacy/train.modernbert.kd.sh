#!/bin/bash -l
#SBATCH --job-name=train-kd
#SBATCH --output=logs/modernbert.out
#SBATCH --error=logs/modernbert.err
#SBATCH --partition=small-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8           # Allocate one gpu per MPI rank
#SBATCH --mem=256G
#SBATCH --time=12:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002438 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

# bsz=64
# nsample=512
bsz=128
nsample=1024
lr=1e-4
split=pos_20.neg_51.filtered
# split=pos_high.neg_zero # 2
# split=pos_half.neg_zero # 3
# split=pos_low.neg_zero # 
# split=pos_high.neg_low #
# split=pos_high.neg_quarter #
# split=pos_zero.neg_high #
max_length=512
model_dir=${HOME}/models/modernbert-crux-researchy-${split}-kd.b${bsz}_n${nsample}.${lr}.${max_length}.1e-5

mkdir -p ${model_dir}

GPUS_PER_NODE=8
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)

# Start experiments
srun singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train_distil \
    --exclude_title \
    --output_dir ${model_dir} \
    --model_name_or_path answerdotai/ModernBERT-base \
    --save_steps 2500 \
    --dataset_name DylanJHJ/crux-researchy-kd \
    --corpus_name DylanJHJ/crux-researchy-corpus \
    --dataset_split $split \
    --per_device_train_batch_size 16 \
    --passage_prefix "search_document: " \
    --query_prefix "search_query: " \
    --train_group_size 8 \
    --bf16 --pooling mean --normalize \
    --temperature 0.02 \
    --distil_temperature 0.1 \
    --prediction_loss_only True \
    --do_eval True \
    --eval_strategy steps \
    --eval_dataset_name DylanJHJ/Qrels \
    --eval_dataset_split msmarco_passage.trec_dl_2019 \
    --eval_corpus_name Tevatron/msmarco-passage-corpus-new \
    --eval_group_size 8 \
    --per_device_eval_batch_size 64 \
    --eval_steps 100 \
    --learning_rate $lr \
    --query_max_len 32 \
    --passage_max_len $max_length \
    --dataloader_num_workers 4 \
    --max_steps 10000 \
    --warmup_steps 1000 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --run_name modernbert-base.crux-researchy-kd-${split}.b${bsz}_n${nsample}.${lr}.${max_length}
