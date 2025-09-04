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
from client import DecentralizedClient, SimpleMNISTModel, constr_prob_matrix
from distributed import run_decentralized_fl

print("Libraries imported")

# Set random seeds for reproducibility and CUDA determinism
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed(SEED)
#    torch.cuda.manual_seed_all(SEED)
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False



def weight_matrix_to_multidigraph(W):
    """
    Convert a weight matrix to a MultiDiGraph.
    W[i][j] can be:
      - a single number → one edge
      - a list of numbers → multiple parallel edges
      - 0/None → no edge
    """
    n = len(W)
    g = nx.MultiDiGraph()
    g.add_nodes_from(range(n))
    
    for i in range(n):
        for j in range(n):
            if W[i][j] is None or W[i][j] == 0:
                continue
            # multiple edges if list
            if isinstance(W[i][j], list):
                for w in W[i][j]:
                    g.add_edge(i, j, weight=w)
            else:
                g.add_edge(i, j, weight=round(W[i][j], 2))
    return g

def display_multidigraph(g, pos):
    plt.figure(figsize=(10, 8))
    #pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos, node_color="skyblue", node_size=500)
    nx.draw_networkx_labels(g, pos, font_size=12)

    # Draw edges with slight curvature
    nx.draw_networkx_edges(
        g, pos,
        edge_color="gray",
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.15"
    )

    # Handle multiple edges for labels
    edge_labels = {}
    for u, v, data in g.edges(data=True):
        label = str(data.get("weight", ""))
        if (u, v) not in edge_labels:
            edge_labels[(u, v)] = [label]
        else:
            edge_labels[(u, v)].append(label)
    
    formatted_labels = {k: ",".join(v) for k, v in edge_labels.items()}

    nx.draw_networkx_edge_labels(
        g, pos,
        edge_labels=formatted_labels,
        font_size=10,
        label_pos=0.5
    )
    plt.axis("off")
    plt.savefig(f"plots/proba_topology.png")





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
    plot_topology(network.G, file_name='original_topology')
    #print("Topology info:\n", OmegaConf.to_yaml(info))
    
    # Prepare the dataset
    data_distributor = DataDistributor(**cfg['dataset'], num_clients=num_clients)
    data_distributor.load_and_distribute()
    summary = data_distributor.get_data_summary()
    plot_heatmap(data_distributor.client_data)
    #print("Data distribution summary:\n", OmegaConf.to_yaml(summary))
    
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
    
    # Plot the new topology with weights
    plot_topology(network.G, Adj=clients[0].A_tilde, file_name='weighted_topology')
    
    
    
    P = constr_prob_matrix(clients)
    print(P)
    
    
    import seaborn as sns
    plt.figure(figsize=(5, 4))
    # Do not display zeros
    mask = P == 0
    sns.heatmap(P, annot=True, cmap="Blues", fmt=".2f", cbar=False, mask=mask)
    plt.title("Transition Probability Matrix")
    plt.xlabel("To Client")
    plt.ylabel("From Client")
    # Save the heatmap
    plt.savefig('plots/transition_probability_matrix.png')
    #plt.show()
    
    #plot_topology(network.G, Adj=clients[0].A_proba, file_name=f'proba_topology(tau={clients[0].tau})')
    # TODO: Check how the training is working
    
    
    # Visualize the MultiDiGraph
    G_proba = weight_matrix_to_multidigraph(P)
    pos = nx.spring_layout(network.G)
    display_multidigraph(G_proba, pos)
    
    
    
    
    
    
    
if __name__ == "__main__":
    
    main()