#!/bin/bash -l
#SBATCH --job-name=labeling
#SBATCH --output=logs/teacherlabel.out.%a
#SBATCH --error=logs/teacherlabel.err.%a
#SBATCH --partition=small-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1           # Allocate one gpu per MPI rank
#SBATCH --mem=256G
#SBATCH --time=24:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002438 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/
source "/users/judylan1/temp/miniconda3/etc/profile.d/conda.sh"
source /users/judylan1/temp/venv/crc/bin/activate

# Start experiments
srun singularity exec $SIF python rerank-for-msmarco.py
