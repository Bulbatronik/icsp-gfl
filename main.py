# Import necessary libraries for decentralized federated learning simulation
import torch

import hydra # Import hydra for configuration management
from omegaconf import DictConfig, OmegaConf

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

import wandb

from topology import NetworkTopology
from visualize import plot_topology, plot_interactive_topology, plot_heatmap
from partitioner import DataDistributor
from client import DecentralizedClient, SimpleMNISTModel
from distributed import run_decentralized_fl

print("Libraries imported")

# Set random seeds for reproducibility and CUDA determinism
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    
    # Initialize Weights & Biases for experiment tracking
    #run = wandb.init(
    #    project="decentralized-federated-learning",
    #    config=OmegaConf.to_container(cfg)
    #)
    num_clients = cfg['network']['num_clients']
   
    # Set the topology
    network = NetworkTopology(**cfg['network'])
    network.create_topology()
    info = network.get_topology_info()
    plot_topology(network.G)
    print("Topology info:\n", OmegaConf.to_yaml(info))
    
    # Prepare the dataset
    data_distributor = DataDistributor(**cfg['dataset'], num_clients=num_clients)
    data_distributor.load_and_distribute()
    summary = data_distributor.get_data_summary()
    plot_heatmap(data_distributor.client_data)
    print("Data distribution summary:\n", OmegaConf.to_yaml(summary))
    
    # Create clients
    clients = {}
    #neighbor_info = {client_id: list(network.G.neighbors(client_id)) for client_id in range(num_clients)}
    for client_id in range(num_clients):
        model = SimpleMNISTModel() # TODO: ADD SELECTION FOR MNIST AND CIFAR10
        train_loader, test_loader = data_distributor.client_loaders[client_id]
        clients[client_id] = DecentralizedClient(
            **cfg['client'],
            client_id=client_id, 
            graph=network.G,
            model=model,
            train_loader=train_loader, 
            test_loader=test_loader, 
            #neighbors=neighbor_info[client_id],
            
        )

    print(clients)
    
    # Check the probabilities
    for client_id, client in clients.items():
        print(f"Client {client_id} neighbors: {list(client.neighbors)}, similarities: {client.neighbors_sim}, probabilities: {client.neighbors_proba}")
        # Select some clients
        selected = client.select_neighbors()
        print(f"Client {client_id} selected neighbors: {selected}")

        selected = client.select_neighbors()
        print(f"Client {client_id} selected neighbors: {selected}")
    
    # TODO: Check how the training is working
    
    
if __name__ == "__main__":
    
    main()