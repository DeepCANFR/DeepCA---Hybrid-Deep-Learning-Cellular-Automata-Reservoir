#!/bin/bash

mkdir logs

CONTROLLER=centralized
for TASK in locomotion jump escape hiking;
do
  # serialized robots
  for i in {0..9}
  do
    sbatch --array=0-9 --nodes=1 -o logs/out.%A_%a.txt -e logs/err.%A_%a.txt brain.sh serialized $i $TASK 0 0 $CONTROLLER
  done
  # random robots
  for i in {0..9}
  do
    sbatch --array=0-9 --nodes=1 -o logs/out.%A_%a.txt -e logs/err.%A_%a.txt brain.sh random $i $TASK 20 20 $CONTROLLER
  done
  # minimize criticality robots
  for i in {0..9}
  do
    sbatch --array=0-9 --nodes=1 -o logs/out.%A_%a.txt -e logs/err.%A_%a.txt brain.sh minimizecriticality $i $TASK 20 20 $CONTROLLER
  done
  # pseudorandom robots
  for i in {0..9}
  do
    sbatch --array=0-9 --nodes=1 -o logs/out.%A_%a.txt -e logs/err.%A_%a.txt brain.sh pseudorandom $i $TASK 20 20 $CONTROLLER
  done
  # box robots
  sbatch --array=0-9 --nodes=1 -o logs/out.%A_%a.txt -e logs/err.%A_%a.txt brain.sh box 0 $TASK 5 4 $CONTROLLER
  # worm robot
  sbatch --array=0-9 --nodes=1 -o logs/out.%A_%a.txt -e logs/err.%A_%a.txt brain.sh worm 0 $TASK 5 4 $CONTROLLER
  # biped robot
  sbatch --array=0-9 --nodes=1 -o logs/out.%A_%a.txt -e logs/err.%A_%a.txt brain.sh biped 0 $TASK 5 4 $CONTROLLER
  # reversedT robot
  sbatch --array=0-9 --nodes=1 -o logs/out.%A_%a.txt -e logs/err.%A_%a.txt brain.sh reversedT 0 $TASK 5 4 $CONTROLLER
done
