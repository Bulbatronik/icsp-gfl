#!/bin/bash

# Activate the environment
source .venv/bin/activate
# Print which python is being used
which python3

echo "=== Starting gradient-based selection experiments for CIFAR10 ==="

# Define arrays for different configuration types
DATASETS=("cifar10_niid") #"cifar10_iid"
NETWORKS=("small" "women")
CLIENT_CONFIG="gradients"
SEEDS=("42" "43" "44")

# Loop through all combinations
for seed in "${SEEDS[@]}"; do
    echo "================ Starting runs for seed=$seed ================"
    for dataset in "${DATASETS[@]}"; do
        for network in "${NETWORKS[@]}"; do
            echo "Running with dataset=$dataset, network=$network, client=$CLIENT_CONFIG" 
            python3 main.py dataset=$dataset client=$CLIENT_CONFIG training=cifar10cnn seed=$seed network=$network 
            # Check the exit status of the Python script
            if [ $? -ne 0 ]; then
                echo "ERROR: The experiment failed with dataset=$dataset, network=$network"
                echo "Pausing execution. Press Enter to continue or Ctrl+C to abort..."
                read -r
            fi
            
            echo "-----------------------------------------" 
        done
    done
done
echo "=== All gradient-based selection experiments for CIFAR10 completed ==="



echo "=== Starting gradient-based selection experiments for FMNIST ==="

# Define arrays for different configuration types
DATASETS=("fmnist_niid") #"cifar10_iid"
NETWORKS=("small" "women")
CLIENT_CONFIG="gradients"
SEEDS=("42" "43" "44")

# Loop through all combinations
for seed in "${SEEDS[@]}"; do
    echo "================ Starting runs for seed=$seed ================"
    for dataset in "${DATASETS[@]}"; do
        for network in "${NETWORKS[@]}"; do
            echo "Running with dataset=$dataset, network=$network, client=$CLIENT_CONFIG" 
            python3 main.py dataset=$dataset client=$CLIENT_CONFIG training=mnistcnn seed=$seed network=$network
            # Check the exit status of the Python script
            if [ $? -ne 0 ]; then
                echo "ERROR: The experiment failed with dataset=$dataset, network=$network"
                echo "Pausing execution. Press Enter to continue or Ctrl+C to abort..."
                read -r
            fi
            
            echo "-----------------------------------------" 
        done
    done
done
echo "=== All gradient-based selection experiments for FMNIST completed ==="
