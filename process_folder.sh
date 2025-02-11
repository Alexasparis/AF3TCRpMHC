#!/bin/bash
#SBATCH --job-name=PF                 # Nombre del job
#SBATCH --time=03:00:00               # Tiempo máximo
#SBATCH --cpus-per-task=20            # Número de CPUs por tarea
#SBATCH --ntasks=1                    # Solo 1 tarea
#SBATCH --output=logs/PF_%j.out       # Salida estándar
#SBATCH --error=logs/PF_%j.err        # Error estándar
#SBATCH --account=bsc72               # Cuenta a usar
#SBATCH -D .                          # Directorio de trabajo
#SBATCH --qos=gp_bscls                # QOS

module purge
module load miniforge
source activate anarci

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

IN_FOLDER=$1                         

if [ ! -d "$IN_FOLDER" ]; then
    echo "Error: Input folder '$IN_FOLDER' does not exist."
    exit 1
fi

mkdir -p ./logs

RESULTS_CSV=./logs/process_folder_job_results.csv

if [ ! -f "$RESULTS_CSV" ]; then
    echo "subfolder,job_id,start,end,elapsed,status" > "$RESULTS_CSV"
fi


for subfolder in "$IN_FOLDER"/tcr_*; do
    [ -e "$subfolder" ] || continue

    START_TIME=$(date +%s)
    
    if python3 process_folder.py "$subfolder" --threshold 70; then
        STATUS="OK"
    else
        STATUS="ERROR"
    fi

    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))

    if date -d "@$START_TIME" +"%Y-%m-%d %H:%M:%S" &>/dev/null; then
        START_TIME_HR=$(date -d "@$START_TIME" +"%Y-%m-%d %H:%M:%S")
        END_TIME_HR=$(date -d "@$END_TIME" +"%Y-%m-%d %H:%M:%S")
    else
        START_TIME_HR=$(date -u -r "$START_TIME" +"%Y-%m-%d %H:%M:%S")
        END_TIME_HR=$(date -u -r "$END_TIME" +"%Y-%m-%d %H:%M:%S")
    fi
    echo "$subfolder,$SLURM_JOB_ID,$START_TIME_HR,$END_TIME_HR,$ELAPSED_TIME,$STATUS" >> "$RESULTS_CSV"
done

