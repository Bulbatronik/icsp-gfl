# Import necessary libraries for decentralized federated learning simulation
import torch

import argparse # Import argparse for command-line argument parsing
import hydra # Import hydra for configuration management
from omegaconf import DictConfig, OmegaConf

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')


from topology import NetworkTopology
from visualize import plot_topology, plot_interactive_topology
from partitioner import DataDistributor
from client import DecentralizedClient
from distributed import run_decentralized_fl

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("Libraries imported")

def merge_config(config, args):
    for arg in vars(args):
        setattr(config, arg, getattr(args, arg))
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Decentralized Federated Learning Simulation")
    parser.add_argument('--num_clients', type=int, help='Number of clients in the network')
    parser.add_argument('--selection_ratio', type=float, help='Ratio of neighbors to select')
    parser.add_argument('--topology', type=str, help='Network topology type (e.g., ring, star, mesh)')
    parser.add_argument('--client_selection', type=str, help='Client selection strategy (e.g., random, embedding, heat)')
    return parser.parse_args()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Merge command-line arguments with configuration file
    args = parse_args()
    cfg = merge_config(cfg, args)
    
    print("Configuration:\n", OmegaConf.to_yaml(cfg))