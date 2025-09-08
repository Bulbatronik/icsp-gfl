#!/bin/bash

# Directory containing network configuration files
NETWORK_DIR="/home/Projects/icsp-gfl/configs/network"

# Get list of network configurations (removing .yaml extension)
networks=($(ls ${NETWORK_DIR} | sed 's/\.yaml$//'))

echo "Found network configurations: ${networks[@]}"
echo "Starting runs for all network configurations..."

# Loop through each network configuration
for network in "${networks[@]}"; do
    echo "================================================================"
    echo "Running with network=${network}"
    echo "================================================================"
    
    # Run the main.py script with the current network configuration
    python3 main.py network=${network}
    
    # Optional: add a small delay between runs if needed
    # sleep 1
done

echo "All network configurations completed."
