# Import necessary libraries for decentralized federated learning simulation
import hydra # Import hydra for configuration management
from omegaconf import DictConfig, OmegaConf

import os
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

import wandb

from topology import NetworkTopology
from visualize import plot_topology, plot_selection_probability, plot_data_distribution, plot_transition_graph
from partitioner import DataDistributor
from client import DecentralizedClient, SimpleMNISTModel, constr_prob_matrix
from distributed import run_decentralized_fl
from utils import set_seed, experiment_name

print("Libraries imported")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg['seed'])
    
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    
    # Initialize Weights & Biases for experiment tracking
    #run = wandb.init(
    #    project="decentralized-federated-learning",
    #    config=OmegaConf.to_container(cfg)
    #)
    
    # Experiment name
    exp_name = f"results/{experiment_name(cfg)}"
    os.makedirs(exp_name, exist_ok=True)
    
    # Get number of clients from config
    num_clients = cfg['network']['num_clients']
   
    # Set the topology
    network = NetworkTopology(**cfg['network'])
    network.create_topology()
    #info = network.get_topology_info()
    #print("Topology info:\n", OmegaConf.to_yaml(info))
    pos = plot_topology(network.G, save_folder=exp_name, file_name='original_topology') # Store the positions of the nodes for better visualization
    
    # Prepare the dataset
    data_distributor = DataDistributor(**cfg['dataset'], num_clients=num_clients, verbose=False)
    data_distributor.load_and_distribute()
    #summary = data_distributor.get_data_summary()
    #print("Data distribution summary:\n", OmegaConf.to_yaml(summary))
    plot_data_distribution(data_distributor.client_data, save_folder=exp_name, file_name='data_distribution')
    
    # Create clients
    model = SimpleMNISTModel() # TODO: ADD SELECTION FOR MNIST AND CIFAR10
    clients = {}
    for client_id in range(num_clients):
        train_loader, test_loader = data_distributor.client_loaders[client_id]
        clients[client_id] = DecentralizedClient(
            **cfg['client'],
            client_id=client_id, 
            graph=network.G,
            model=deepcopy(model) ,
            train_loader=train_loader, 
            test_loader=test_loader, 
        )
    #print(clients)
    
    # Check the probabilities # REMOVE LATER
    #for client_id, client in clients.items():
    #    print(f"Client {client_id} neighbors: {list(client.neighbors)}, similarities: {client.neighbors_sim}, probabilities: {client.neighbors_proba}")
    #    # Select some clients
    #    selected = client.select_neighbors()
    #    print(f"Client {client_id} selected neighbors: {selected}")
    #    selected = client.select_neighbors()
    #    print(f"Client {client_id} selected neighbors: {selected}")
    
    # Plot the new topology with weights
    plot_topology(network.G, Adj=clients[0].A_tilde, save_folder=exp_name, file_name='weighted_topology')
    # Plot the selection probability matrix
    P = constr_prob_matrix(clients)
    plot_selection_probability(P, save_folder=exp_name, file_name = 'selection_probability_matrix')
    # Plot the graph where edges represent the selection probability
    plot_transition_graph(P, pos, save_folder=exp_name, file_name='selection_graph')
    
    
    # TODO: Try training
    
if __name__ == "__main__":
    
    main()