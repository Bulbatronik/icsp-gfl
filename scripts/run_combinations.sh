#!/bin/bash

# Define arrays for different configuration types
DATASETS=("mnist_iid" "mnist_niid")
CLIENT_CONFIGS=("random" "broadcast")

# Loop through all combinations
for dataset in "${DATASETS[@]}"; do
    for client_config in "${CLIENT_CONFIGS[@]}"; do
        echo "Running with dataset=$dataset and client=$client_config"
        python3 main.py dataset=$dataset client=$client_config
        echo "-----------------------------------------"
    done
done
echo "=== All combinations completed ==="
