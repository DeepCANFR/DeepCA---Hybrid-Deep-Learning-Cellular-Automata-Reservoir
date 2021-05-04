#!/bin/bash
#SBATCH --time=2-23:00
#SBATCH --partition=slowq

DIRNAME=sr4rc.$SLURM_ARRAY_JOB_ID.body
mkdir $DIRNAME
module load java/jdk/14.0.2

#evolution
/cm/shared/apps/java/jdk/14.0.2/bin/java -cp body.jar it.units.erallab.BodyOptimization randomSeed=$SLURM_ARRAY_TASK_ID gridSize=10 robotVoxels=20 popSize=1000 iterations=200 dir=$DIRNAME statsFile=stats-$SLURM_ARRAY_TASK_ID.txt

# SCHEDULE: sbatch --array=0-10 --nodes=1 -o logs/out.%A_%a.txt -e logs/err.%A_%a.txt body.sh
# STATUS: squeue -u $USER
# CANCEL: scancel
