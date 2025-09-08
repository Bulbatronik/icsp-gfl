#!/bin/bash

# Run with config_eucl.yaml
python main.py --config-name=config_eucl client.tau=1.5 # Overwrite tau

# Run with config_cosine.yaml
python main.py --config-name=config_cosine client.tau=1.5 # Overwrite tau
