# Import necessary libraries for decentralized federated learning simulation
import os
from collections import Counter
from typing import Dict
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings('ignore')

# Increase the quality
plt.rcParams['figure.dpi'] = 150


# Visualization Functions for Network Topologies
def plot_topology(G: nx.Graph, layout_type: str = 'spring', Adj: np.ndarray = None, 
                  save_folder: str = './', file_name: str = 'original_topology') -> None:
    """Plot a network topology using matplotlib"""
    plt.figure(figsize=(10, 8))
    
    # Choose layout
    if layout_type == 'spring':
        pos = nx.spring_layout(G)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'shell':
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Draw the network
    if Adj is not None: # Colorize the edges
        G = nx.from_numpy_array(Adj)
        # Get edge weights for coloring
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

        # Normalize weights for colormap
        norm = plt.Normalize(vmin=min(weights), vmax=max(weights))
        cmap = plt.cm.coolwarm
        edge_colors = [cmap(norm(w)) for w in weights]


        # Show edge similarity values alongside colored links
        for (i, j), w, color in zip(edges, weights, edge_colors):
                x0, y0 = pos[i]
                x1, y1 = pos[j]
                plt.plot([x0, x1], [y0, y1], color=color, linewidth=4, alpha=0.9)
                # Place similarity value at midpoint of edge
                xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
                plt.text(xm, ym, f"{w:.2f}", color='black', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))    
    
        # Redraw nodes and labels on top
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    else:
        # Draw nodes and edges separately instead of using nx.draw() to avoid title issues
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add title after drawing the network
    plt.title("Network of Clients", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    # Create the figure
    save_path = f'{save_folder}/plots'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{file_name}.png')
    return pos

    
def plot_data_distribution(client_data: Dict[int, Dict], save_folder: str = './', file_name: str = 'data_distribution') -> None:
        """Plot heatmap of label distribution across clients"""
        
        num_clients = len(client_data)
        # Prepare data
        label_counts = np.zeros((num_clients, 10), dtype=int)
        
        for client_id in range(num_clients):
            train_labels = [int(label) for label in client_data[client_id]['train_dataset'].dataset.targets[client_data[client_id]['train_indices']]]
            counts = Counter(train_labels)
            for label, count in counts.items():
                label_counts[client_id, label] = count
        
        # Create DataFrame for seaborn
        df = pd.DataFrame(label_counts, columns=[f'{i}' for i in range(10)], index=[f'{i}' for i in range(num_clients)])
        
        plt.figure(figsize=(10, 6))
        # Fix heatmap to be from 0 to total_labels/num_clients
        sns.heatmap(df, annot=True, fmt='d', cmap='YlGnBu', vmin=0, vmax=7000)
        plt.title('Label Distribution Across Clients', fontsize=16)
        plt.xlabel('Class ID', fontsize=14)
        plt.ylabel('Client ID', fontsize=14)
        # Adjust layout and save
        plt.tight_layout()
        # Create the figure
        save_path = f'{save_folder}/plots'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{file_name}.png')
    
def plot_selection_probability(P: np.ndarray, save_folder: str = './', file_name: str = 'selection_probability_matrix'):
    plt.figure(figsize=(5, 4))
    # Do not display zeros
    mask = P == 0
    sns.heatmap(P, annot=True, cmap="Blues", fmt=".2f", cbar=False, mask=mask)
    plt.title("Transition Probability Matrix", fontsize=10)
    plt.xlabel("To Client")
    plt.ylabel("From Client")
    plt.tight_layout()
    # Create the figure
    save_path = f'{save_folder}/plots'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{file_name}.png')


def plot_transition_graph(P: np.ndarray, pos: dict, save_folder: str = './', file_name: str = 'selection_graph'):
        """
        Creates and plots a directed graph from an asymmetrical, bidirectional
        transition probability matrix.

        The edges are colored based on their probability values using a heatmap.
        A color bar is included to serve as a legend for the edge colors.
        Bidirectional edges are drawn with a curve to show both directions clearly.

        Args:
            P (np.ndarray): An asymmetrical transition probability matrix.
            pos (dict): A dictionary mapping node labels to their (x, y) coordinates
                        for plotting. Example: {0: (0, 0), 1: (1, 1), ...}.
            filename (str): The name of the file to save the plot.

        Raises:
            ValueError: If the input matrix is not a square matrix.
        """
        # Check if the matrix is square
        if P.shape[0] != P.shape[1]:
            raise ValueError("The input matrix must be square.")
        # Create a directed graph
        G = nx.DiGraph()
        # Add nodes and edges from the matrix
        num_nodes = P.shape[0]
        for i in range(num_nodes):
            for j in range(num_nodes):
                # Add a directed edge if the transition probability is greater than 0
                prob = P[i, j]
                if prob > 0:
                    G.add_edge(i, j, weight=prob)
        # Extract edge weights for coloring
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        # Normalize the edge weights to a [0, 1] range for the colormap
        # Handle the case where all weights are the same to avoid division by zero
        if not edge_weights or max(edge_weights) == min(edge_weights):
            norm = mcolors.Normalize(vmin=0, vmax=1)
        else:
            norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        # Choose a colormap and map the weights to colors
        cmap = cm.viridis
        edge_colors = [cmap(norm(weight)) for weight in edge_weights]

        # Create the plot
        plt.figure(figsize=(10, 8))
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')#, alpha=0.9)
        # Draw edges with the calculated colors and widths
        # The 'edge_color' parameter accepts a list of colors
        nx.draw_networkx_edges(G, pos,
                            edge_color=edge_colors,
                            width=2.5,
                            alpha=0.8,
                            arrows=True,
                            arrowstyle='->',
                            arrowsize=20,
                            connectionstyle='arc3,rad=0.1')
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

        # Create a ScalarMappable object for the color bar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(edge_weights)
        # Add the color bar to the plot
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Selection Probability')

        plt.title('Selection Probability Graph', fontsize=16)
        plt.axis('off')  # Turn off the axis
        plt.tight_layout()
        # Create the figure
        save_path = f'{save_folder}/plots'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{file_name}.png')