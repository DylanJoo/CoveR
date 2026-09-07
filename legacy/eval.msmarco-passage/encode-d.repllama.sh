#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=m-enc-doc.out
#SBATCH --error=m-enc-doc.err
#SBATCH --partition=dev-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=8                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8           # Allocate one gpu per MPI rank
#SBATCH --gpus-per-task=8           # Allocate one gpu per MPI rank
#SBATCH --mem=256G
#SBATCH --time=0-02:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

model_dir=${HOME}/models/repllama-msmarco-psg.b128_n512.1e-4
checkpoint=checkpoint-4000
output_dir=${HOME}/indices/msmarco-passage-corpus-new/${model_dir##*/}
mkdir -p $output_dir

NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
TOTAL_SHARDS=64 # total shards (64 shards = 8 nodes * 8 GPUs)
SHARDS_PER_NODE=$(( TOTAL_SHARDS / SLURM_NNODES ))
EXCLUDE_TITLE=true

i=0
for node in $NODELIST; do
  echo ">>> Launching encoding on node: $node"

  # Calculate shard range for this node
  START=$(( i * SHARDS_PER_NODE ))
  END=$(( START + SHARDS_PER_NODE - 1 ))

  echo "Node $node will handle shards $START–$END"

  # Launch the job on that specific node
  srun --nodes=1 --ntasks=1 --nodelist=$node bash -c "
    for gpu in {0..7}; do
      SHARD=\$(( $START + gpu ))
      export CUDA_VISIBLE_DEVICES=\$gpu
      export HIP_VISIBLE_DEVICES=\$gpu
      echo \"Node $node, GPU \$gpu → Shard \$SHARD\"

      singularity exec $SIF \
        python -m tevatron.retriever.driver.encode \
          --output_dir=temp \
          --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
          --lora_name_or_path $model_dir/$checkpoint \
          --lora \
          --bf16 \
          ${EXCLUDE_TITLE:+--exclude_title} \
          --per_device_eval_batch_size 200 \
          --normalize \
          --pooling last \
          --passage_prefix \"passage: \" \
          --append_eos_token \
          --passage_max_len 256 \
          --dataset_number_of_shards $TOTAL_SHARDS \
          --corpus_name Tevatron/msmarco-passage-corpus-new \
          --encode_output_path $output_dir/corpus_emb.\${SHARD}.pkl \
          --dataset_shard_index \${SHARD} & 
    done
    wait
  " &
  ((i++))
done

wait
echo "Done"
