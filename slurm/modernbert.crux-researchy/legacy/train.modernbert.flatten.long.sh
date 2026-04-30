#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=logs/modernbert.out.long
#SBATCH --error=logs/modernbert.err.long
#SBATCH --partition=small-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002438 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

bsz=32
nsample=256
lr=1e-4
split=flatten.pos_5.neg_1
max_length=1024
model_dir=${HOME}/models/modernbert-crux-researchy-${split}.b${bsz}_n${nsample}.${lr}.${max_length}

mkdir -p ${model_dir}

GPUS_PER_NODE=4
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)

# Start experiments
srun singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train \
    --exclude_title \
    --output_dir ${model_dir} \
    --model_name_or_path answerdotai/ModernBERT-base \
    --save_steps 5000 \
    --dataset_name DylanJHJ/crux-researchy \
    --corpus_name DylanJHJ/crux-researchy-corpus \
    --dataset_split $split \
    --per_device_train_batch_size 8 \
    --train_group_size 8 \
    --pooling mean --normalize \
    --temperature 0.02 \
    --prediction_loss_only True \
    --do_eval True \
    --eval_strategy steps \
    --eval_dataset_name DylanJHJ/Qrels \
    --eval_dataset_split msmarco_passage.trec_dl_2019 \
    --eval_corpus_name Tevatron/msmarco-passage-corpus-new \
    --eval_group_size 8 \
    --per_device_eval_batch_size 64 \
    --eval_steps 100 \
    --bf16 \
    --learning_rate $lr \
    --query_max_len 32 \
    --passage_max_len $max_length \
    --dataloader_num_workers 4 \
    --max_steps 50000 \
    --warmup_steps 5000 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --run_name modernbert-base.crux-researchy-${split}.b${bsz}_n${nsample}.${lr}.${max_length}
