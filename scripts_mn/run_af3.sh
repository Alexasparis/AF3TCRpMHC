#!/bin/bash
#SBATCH --job-name=AF3
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres gpu:1
#SBATCH --ntasks=1
#SBATCH --output=logs/AF3_%j.out
#SBATCH --error=logs/AF3_%j.err
#SBATCH --account=bsc72
#SBATCH -D .
#SBATCH --qos=acc_bscls
##SBATCH --qos=acc_bscls

module purge
module load singularity cuda/12.6
module load alphafold/3.0.0
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

IN_FOLDER_WITH_JSONS=$1
OUT_FOLDER=$(pwd)
WEIGHTS=/gpfs/projects/bsc72/weights/AF3/

# Wrapper inputs:
#    - fasta path
#    - output dir
#    - weights

# Working
#bsc_alphafold /gpfs/projects/bsc72/annadiaz/test-af3/test_json/ /gpfs/projects/bsc72/annadiaz/test-af3 /gpfs/projects/bsc72/weights/AF3/

bsc_alphafold $IN_FOLDER_WITH_JSONS $OUT_FOLDER $WEIGHTS
