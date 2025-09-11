#!/bin/bash

# Script to set up and run the multiprocessing version of the code

echo "====== Parallel Federated Learning Execution ======"
echo "Using multiprocessing implementation (client_mp.py, distributed_mp.py, main_mp.py)"
echo "======================================================"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found in current directory."
fi

# Check if tqdm is installed (for progress bars)
pip install tqdm

# Run the multiprocessing version
python main_mp.py "$@"