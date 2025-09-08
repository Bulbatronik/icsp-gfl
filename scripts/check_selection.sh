#!/bin/bash

tau=1.5
# Define an array of t values to test
t_values=(0.1 5 10 50 100 300)

# Run with config_spectr_eucl.yaml
#echo "Running with config_spectr_eucl.yaml"
#python main.py --config-name=config_spectr_eucl client.tau=$tau # Overwrite tau

# Run with config_spectr_cosine.yaml
#echo "Running with config_spectr_cosine.yaml"
#python main.py --config-name=config_spectr_cosine client.tau=$tau # Overwrite tau

# Run with config_heatkernel for each value of t
echo "Running with config_heatkernel.yaml for multiple t values"
for t in "${t_values[@]}"; do
    echo "Running with t=$t"
    python main.py --config-name=config_heatkernel client.t=$t
done
