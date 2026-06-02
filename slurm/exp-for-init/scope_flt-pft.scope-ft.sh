#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=logs/pft/out
#SBATCH --error=logs/pft/err
#SBATCH --partition=small-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --array=0
#SBATCH --mem=128G
#SBATCH --time=4:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002438 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/
export TOKENIZERS_PARALLELISM=false
export CRUX_ROOT=${HOME}/datasets/crux

# Models
model_dir=${HOME}/models/cov-contrastive/scope_flt-pft.10k
mkdir -p ${model_dir}

GPUS_PER_NODE=4
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)
PRETRAINED=DylanJHJ/nomic.modernbert-base.scope-flatten.10k
SPLIT=pos_half.neu_low.neg_zero

# Start experiments
srun singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train_dualdistil \
    --exclude_title \
    --output_dir ${model_dir} \
    --model_name_or_path $PRETRAINED \
    --save_steps 1000 \
    --dataset_name DylanJHJ/crux-researchy-kdnew-ext \
    --corpus_name DylanJHJ/crux-researchy-corpus \
    --dataset_split $SPLIT \
    --per_device_train_batch_size 16 \
    --train_group_size 8 \
    --prediction_loss_only True \
    --bf16 --pooling mean --normalize \
    --passage_prefix "search_document: " \
    --query_prefix "search_query: " \
    --subquery_prefix "search_query: " \
    --temperature 0.02 \
    --use_crossentropy 1.0 \
    --use_kld 0.0 \
    --contrastive_lambda 1.0 \
    --sq_contrastive_lambda 1.0 \
    --covdistil_method KLD \
    --covdistil_lambda 0.0 \
    --eval_steps 1000 \
    --learning_rate 1e-4 \
    --query_max_len 180 \
    --passage_max_len 512 \
    --dataloader_num_workers 8 \
    --lr_scheduler_type 'cosine' \
    --max_steps 10000 \
    --warmup_steps 500 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --run_name ${model_dir##*/}
