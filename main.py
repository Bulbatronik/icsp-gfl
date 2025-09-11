# Import necessary libraries for decentralized federated learning simulation
import hydra # Import hydra for configuration management
from omegaconf import DictConfig, OmegaConf
import os
import torch
from copy import deepcopy
import warnings
import wandb
from topology import NetworkTopology
from visualize import plot_topology, plot_selection_probability, plot_data_distribution, plot_transition_graph
from partitioner import DataDistributor
from models import load_model
from client import DecentralizedClient, constr_prob_matrix
from distributed import run_decentralized_fl
from utils import set_seed, experiment_name

warnings.filterwarnings('ignore')
print("Libraries imported")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg['seed'])
    
    # Check if device is specified in client config, default to CUDA if available
    requested_device = cfg['federation'].get('device', 'cuda')
    if requested_device == 'cuda' and not torch.cuda.is_available():
        print("\033[93mWarning: CUDA requested but not available. Falling back to CPU.\033[0m")
        device = torch.device('cpu')
    else:
        device = torch.device(requested_device)
    
    
    # Clear cuda cache if using CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    
    run = wandb.init(
        project="decentralized-federated-learning",
        name=experiment_name(cfg),
        # Don't specify entity to use default personal space
        config=OmegaConf.to_container(cfg)
    )
    
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    print(f"Using device: {device}")
    
    # Set the topology
    network = NetworkTopology(**cfg['network'])
    network.create_topology()
    
    # Change the number of clients (for cases like "karate club", etc.)
    num_clients = network.num_clients
    cfg['network']['num_clients'] = num_clients
    
    # Experiment name where the figures will be saved
    exp_name = f"results/{experiment_name(cfg)}"
    os.makedirs(exp_name, exist_ok=True)
    

    #info = network.get_topology_info()
    #print("Topology info:\n", OmegaConf.to_yaml(info))
    pos = plot_topology(network.G, save_folder=exp_name, file_name='original_topology') # Store the positions of the nodes for better visualization
    
    # Create a dictionary to collect all initial visualizations
    initial_visuals = {}
    
    # Add the topology figure to the collection
    initial_visuals["original_topology"] = wandb.Image(f"{exp_name}/plots/original_topology.png")
    
    # Prepare the dataset
    data_distributor = DataDistributor(**cfg['dataset'], num_clients=num_clients, verbose=False)
    data_distributor.load_and_distribute()
    #summary = data_distributor.get_data_summary()
    #print("Data distribution summary:\n", OmegaConf.to_yaml(summary))
    plot_data_distribution(data_distributor.client_data, save_folder=exp_name, file_name='data_distribution')
    
    # Add the data distribution figure to the collection
    initial_visuals["data_distribution"] = wandb.Image(f"{exp_name}/plots/data_distribution.png")
    
    # Create clients
    #model = SimpleMNISTModel(device=device) # TODO: ADD MODEL SELECTION
    model = load_model(cfg['client']['name'], device=device)
    
    clients = {}
    for client_id in range(num_clients):
        loaders = data_distributor.client_loaders[client_id]
        clients[client_id] = DecentralizedClient(
            **cfg['client'],
            client_id=client_id, 
            graph=network.G,
            model=deepcopy(model),
            train_loader=loaders["train_loader"], 
            test_loader=loaders["test_loader"],
            device=device  # Pass the device to the client
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
    plot_topology(network.G, Adj=clients[0].A_tilde, pos=pos, save_folder=exp_name, file_name='weighted_topology')
    # Plot the selection probability matrix
    P = constr_prob_matrix(clients)
    plot_selection_probability(P, save_folder=exp_name, file_name = 'selection_probability_matrix')
    # Plot the graph where edges represent the selection probability
    plot_transition_graph(P, pos, save_folder=exp_name, file_name='selection_graph')
    
    # Log these visualizations to wandb
    wandb.log({
        "weighted_topology": wandb.Image(f"{exp_name}/plots/weighted_topology.png"),
        "selection_probability_matrix": wandb.Image(f"{exp_name}/plots/selection_probability_matrix.png"),
        "selection_graph": wandb.Image(f"{exp_name}/plots/selection_graph.png")
    })
    
    
    # Run decentralized federated learning
    print("Starting decentralized federated learning...")
    results = run_decentralized_fl(
        **cfg['federation'],
        clients=clients,
    )
    
    # Save the results to a file in the experiment directory
    import json
    with open(f"{exp_name}/results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Close wandb run
    wandb.finish()
    
    print(f"Training completed. Results saved to {exp_name}/results.json")
    
if __name__ == "__main__":
    
    main()