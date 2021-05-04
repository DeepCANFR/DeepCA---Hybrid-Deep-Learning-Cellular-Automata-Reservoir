#!/bin/bash
#SBATCH --partition=slowq

JAVA=java/jdk/14.0.2

module load $JAVA

BODY=$1
ROBOT=$2
TASK=$3
W=$4
H=$5
CONTROLLER=$6
DIRNAME=sr4rc.$SLURM_ARRAY_JOB_ID.brain.$BODY.$ROBOT.$TASK
mkdir $DIRNAME

#evolution
/cm/shared/apps/${JAVA}/bin/java -cp brain.jar it.units.erallab.ControllerOptimization randomSeed=${SLURM_ARRAY_JOB_ID}${SLURM_ARRAY_TASK_ID} bodyType=$BODY robotIndex=$ROBOT taskType=$TASK gridW=$W gridH=$H controller=$CONTROLLER robotVoxels=20 births=10000 dir=$DIRNAME statsFile=stats-$SLURM_ARRAY_TASK_ID.txt

# SCHEDULE: sbatch --array=0-9 --nodes=1 -o logs/out.%A_%a.txt -e logs/err.%A_%a.txt brain.sh
# STATUS: squeue -u $USER
# CANCEL: scancel
