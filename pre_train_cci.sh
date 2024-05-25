#!/bin/bash

# Assign variables
TASK=$1
MODEL=$2
DATASET=$3
CONFIG_FILE=$4

# Define the task command
TASK_CMD="python3 ./run_pretrain.py --task ${TASK} --model ${MODEL} --dataset ${DATASET} --gpu_id 0 --config_file ${CONFIG_FILE}"

# Execute the task
echo "Running command: ${TASK_CMD}"
eval ${TASK_CMD}

