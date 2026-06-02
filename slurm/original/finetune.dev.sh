#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=logs/debug.out
#SBATCH --error=logs/debug.err
#SBATCH --partition=dev-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2           # Allocate one gpu per MPI rank
#SBATCH --mem=256G
#SBATCH --time=2:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002438 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/
export TOKENIZERS_PARALLELISM=false

bsz=64
nsample=512
lr=1e-4
PRETRAINED=DylanJHJ/nomic.modernbert-base.msmarco-passage.10k
AGG="mean"

split=pos_half.neg_zero
model_dir=${HOME}/models/dev

rm -r ${model_dir}
mkdir -p ${model_dir}

GPUS_PER_NODE=2
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)

# Start experiments
srun singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train_covdistil \
    --exclude_title \
    --output_dir ${model_dir} \
    --model_name_or_path $PRETRAINED \
    --save_steps 10000 \
    --dataset_name DylanJHJ/crux-researchy-new \
    --corpus_name DylanJHJ/crux-researchy-corpus \
    --dataset_split $split \
    --per_device_train_batch_size 16 \
    --train_group_size 8 \
    --prediction_loss_only True \
    --bf16 --pooling mean --normalize \
    --passage_prefix "search_document: " \
    --query_prefix "search_query:[unused0][unused1][unused2][unused3][unused4]" \
    --num_views 5 \
    --view_pooling 'cluster' \
    --subquery_prefix "search_query: " \
    --request_as_query \
    --temperature 0.02 \
    --distil_temperature 0.02 \
    --aggregation_strategy $AGG \
    --covdistil_method KLD \
    --covdistil_lambda 0.1 \
    --sq_contrastive_lambda 0.1 \
    --view_orthogonalize_lambda 0.1 \
    --view_orthogonalize_method pairwise \
    --eval_steps 10 \
    --learning_rate $lr \
    --query_max_len 128 \
    --passage_max_len 512 \
    --dataloader_num_workers 4 \
    --lr_scheduler_type 'cosine' \
    --weight_decay 0.01 \
    --max_steps 200 \
    --warmup_steps 50 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --run_name ${model_dir##*/}
