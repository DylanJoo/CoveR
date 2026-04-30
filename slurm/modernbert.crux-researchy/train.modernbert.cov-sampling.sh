#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=logs/cov-sampling.out.%a
#SBATCH --error=logs/cov-sampling.err.%a
#SBATCH --partition=small-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --array=0-6
#SBATCH --mem=256G
#SBATCH --time=12:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002438 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/
export TOKENIZERS_PARALLELISM=false

bsz=64
nsample=512
lr=1e-4
SPLIT=(
"pos_20.neg_51"
"pos_high.neg_zero"
"pos_half.neg_zero"
"pos_low.neg_zero"
"pos_high.neg_low"
"pos_high.neg_quarter"
"pos_zero.neg_high"
)
split=${SPLIT[$SLURM_ARRAY_TASK_ID]}

model_dir=${HOME}/models/ablation.cov-sampling/modernbert-crux-researchy-${split}.b${bsz}_n${nsample}.${lr}.request
mkdir -p ${model_dir}

GPUS_PER_NODE=4
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)
PRETRAINED=nomic-ai/modernbert-embed-base-unsupervised

# Start experiments
srun singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train_dev \
    --exclude_title \
    --output_dir ${model_dir} \
    --model_name_or_path $PRETRAINED \
    --save_steps 1000 \
    --dataset_name DylanJHJ/crux-researchy-new \
    --corpus_name DylanJHJ/crux-researchy-corpus \
    --dataset_split $split \
    --per_device_train_batch_size 16 \
    --train_group_size 8 \
    --prediction_loss_only True \
    --bf16 --pooling mean --normalize \
    --passage_prefix "search_document: " \
    --query_prefix "search_query: " \
    --request_as_query \
    --temperature 0.02 \
    --eval_steps 1000 \
    --learning_rate $lr \
    --query_max_len 128 \
    --passage_max_len 512 \
    --dataloader_num_workers 4 \
    --lr_scheduler_type 'cosine' \
    --weight_decay 0.01 \
    --max_steps 5000 \
    --warmup_steps 500 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --run_name ${model_dir##*/}
