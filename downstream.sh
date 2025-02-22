#!/bin/bash

# Assign variables
TASK=$1
MODEL=$2
DATASET=$3
CONFIG_FILE=$4


TASK_CMD="python3 ./downstream_config.py --dataname "$dataname" --gnn_type "$gnn_type" --pre_train "$pre_train" --epoch_num $epoch_num --num_class $num_class --config_num $config_num > "logs/$log_file" 2>&1 "
# Execute the task
echo "Running command: ${TASK_CMD}"
eval ${TASK_CMD}

# Define the task command
TASK_CMD="python3 ./run_pretrain.py --task ${TASK} --model ${MODEL} --dataset ${DATASET} --gpu_id 0 --config_file ${CONFIG_FILE}"
# Define parameter arrays
datanames=("CiteSeer") #"Cora"
gnn_types=("GCN")
pre_trains=("GraphCL" "SimGRACE")
epoch_nums=(50 100 500 1000)
num_classes=(6 7)

# Loop through combinations and execute the Python script with parameters
for index in ${!datanames[@]};
do
    dataname=${datanames[$index]}
    num_class=${num_classes[$index]}
    for gnn_type in "${gnn_types[@]}"
    do
        for pre_train in "${pre_trains[@]}"
        do
            for epoch_num in "${epoch_nums[@]}"
            do
                for config_num in {4..19}  # Loop from 0 to 19
                do
                    # Construct a unique log file name that includes config_num
                    log_file="${dataname}_${gnn_type}_${pre_train}_${epoch_num}_${num_class}_config${config_num}.log"
                    echo "Running model with $dataname, $gnn_type, $pre_train, $epoch_num, $num_class, config_num $config_num"
                    # Execute the python script and redirect output to log file
                    singularity exec --nv /data/qianMa/SIF/scaling.sif python3 ./downstream_config.py --dataname "$dataname" --gnn_type "$gnn_type" --pre_train "$pre_train" --epoch_num $epoch_num --num_class $num_class --config_num $config_num > "logs/$log_file" 2>&1
                done
            done
        done
    done
done
