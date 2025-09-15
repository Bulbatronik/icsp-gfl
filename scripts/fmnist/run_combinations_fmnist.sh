#!/bin/bash

# Activate the enviromnent
source .venv/bin/activate
# Print which python is being used
which python3

echo "=== Starting runs for all dataset and client configuration combinations ==="

# Define arrays for different configuration types
DATASETS=("fmnist_iid" "fmnist_niid")
CLIENT_CONFIGS=("nofed" "random" "broadcast" "spect_eig3_cos")
SEEDS=("42" "43" "44")
# Loop through all combinations
for seed in "${SEEDS[@]}"; do
    echo ================" Starting runs for seed=$seed ================"
    for dataset in "${DATASETS[@]}"; do
        for client_config in "${CLIENT_CONFIGS[@]}"; do
            echo "Running with dataset=$dataset and client=$client_config"
            python3 main.py dataset=$dataset client=$client_config training=mnistcnn seed=$seed
            
            # Check the exit status of the Python script
            if [ $? -ne 0 ]; then
                echo "ERROR: The experiment failed with dataset=$dataset and client=$client_config"
                echo "Pausing execution. Press Enter to continue or Ctrl+C to abort..."
                read -r
            fi
            
            echo "-----------------------------------------" 
        done
    done
done
echo "=== All combinations completed ==="
