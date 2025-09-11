# Import necessary libraries for decentralized federated learning simulation
import os
from collections import Counter
from typing import Dict, Optional
import networkx as nx
import torch
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

class AdaptiveVisualizer:
    """
    A class to create adaptive visualizations for network graphs of different sizes and topologies.
    The visualizations automatically adjust settings based on the graph properties.
    """
    
    def __init__(self, save_folder: str = './'):
        """
        Initialize the AdaptiveVisualizer with default settings.
        
        Args:
            save_folder (str): Directory where visualizations will be saved
        """
        self.save_folder = save_folder
        self.save_path = f'{save_folder}/plots'
        os.makedirs(self.save_path, exist_ok=True)
        
        # Default color schemes
        self.node_color_scheme = 'lightblue'
        self.edge_color_scheme = 'gray'
        self.edge_cmap = plt.cm.coolwarm
        self.heatmap_cmap = 'YlGnBu'
        self.transition_cmap = cm.viridis
        
    def _calculate_adaptive_settings(self, G: nx.Graph) -> Dict:
        """
        Calculate adaptive settings based on graph properties.
        
        Args:
            G (nx.Graph): The graph to be visualized
            
        Returns:
            Dict: Dictionary of adaptive settings
        """
        num_nodes = G.number_of_nodes()
        
        # Calculate adaptive node size based on number of nodes
        if num_nodes <= 10:
            node_size = 1000
            font_size = 12
            edge_width = 2.5
            figure_size = (10, 8)
        elif num_nodes <= 20:
            node_size = 800
            font_size = 10
            edge_width = 2.0
            figure_size = (12, 10)
        elif num_nodes <= 50:
            node_size = 500
            font_size = 8
            edge_width = 1.5
            figure_size = (14, 12)
        elif num_nodes <= 100:
            node_size = 300
            font_size = 6
            edge_width = 1.0
            figure_size = (16, 14)
        else:
            node_size = 100
            font_size = 0  # No labels for very large graphs
            edge_width = 0.5
            figure_size = (18, 16)
            
        # Adjust alpha based on density
        density = nx.density(G)
        edge_alpha = max(0.3, min(0.9, 1.0 - density))
        
        return {
            'node_size': node_size,
            'font_size': font_size,
            'edge_width': edge_width,
            'figure_size': figure_size,
            'edge_alpha': edge_alpha
        }
    
    def _get_best_layout(self, G: nx.Graph, layout_type: str = 'auto') -> Dict:
        """
        Choose the best layout based on graph properties or user preference.
        
        Args:
            G (nx.Graph): The graph to be visualized
            layout_type (str): Type of layout requested or 'auto' for automatic selection
            
        Returns:
            Dict: Node positions for the graph
        """
        num_nodes = G.number_of_nodes()
        density = nx.density(G)
        
        if layout_type == 'auto':
            # Choose layout based on graph properties
            if num_nodes <= 20:
                if density > 0.5:
                    layout_type = 'circular'
                else:
                    layout_type = 'spring'
            elif num_nodes <= 50:
                if density > 0.3:
                    layout_type = 'circular'
                else:
                    layout_type = 'spring'
            else:
                layout_type = 'spring'  # Default for large graphs
        
        # Apply the selected layout
        if layout_type == 'spring':
            pos = nx.spring_layout(G, k=1/np.sqrt(num_nodes), iterations=100)
        elif layout_type == 'circular':
            pos = nx.circular_layout(G)
        elif layout_type == 'shell':
            pos = nx.shell_layout(G)
        elif layout_type == 'spectral':
            pos = nx.spectral_layout(G)
        elif layout_type == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:  # Default
            pos = nx.spring_layout(G)
            
        return pos
    
    def plot_topology(self, G: nx.Graph, layout_type: str = 'auto', Adj: Optional[np.ndarray] = None, 
                     pos: Optional[Dict] = None, file_name: str = 'adaptive_topology') -> Dict:
        """
        Plot a network topology with adaptive settings based on graph size.
        
        Args:
            G (nx.Graph): The graph to be visualized
            layout_type (str): Type of layout to use ('auto', 'spring', 'circular', etc.)
            Adj (np.ndarray, optional): Adjacency matrix for edge weights
            pos (Dict, optional): Predefined node positions
            file_name (str): Name for the saved file
            
        Returns:
            Dict: Node positions used in the visualization
        """
        # Get adaptive settings
        settings = self._calculate_adaptive_settings(G)
        
        # Create figure with adaptive size
        plt.figure(figsize=settings['figure_size'])
        
        # Get node positions
        if pos is None:
            pos = self._get_best_layout(G, layout_type)
        
        if Adj is not None:
            # Draw weighted edges
            G_weighted = nx.from_numpy_array(Adj)
            
            # Get edge weights
            edge_attrs = nx.get_edge_attributes(G_weighted, 'weight')
            if edge_attrs:
                edges, weights = zip(*edge_attrs.items())
                
                # Normalize weights for colormap
                if len(set(weights)) > 1:
                    norm = plt.Normalize(vmin=min(weights), vmax=max(weights))
                else:
                    norm = plt.Normalize(vmin=0, vmax=1)
                    
                edge_colors = [self.edge_cmap(norm(w)) for w in weights]
                
                # Draw edges with weights
                for (i, j), w, color in zip(edges, weights, edge_colors):
                    x0, y0 = pos[i]
                    x1, y1 = pos[j]
                    plt.plot([x0, x1], [y0, y1], color=color, linewidth=settings['edge_width'], alpha=settings['edge_alpha'])
                    
                    # Only show weight labels if there are few nodes and the font is readable
                    if settings['font_size'] >= 8:
                        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
                        bbox_props = dict(
                            facecolor='white', 
                            edgecolor='none', 
                            alpha=0.7, 
                            boxstyle='round,pad=0.2',
                            mutation_scale=max(5, settings['font_size'] * 0.8)
                        )
                        plt.text(xm, ym, f"{w:.2f}", color='black', 
                                fontsize=max(6, settings['font_size'] - 2), 
                                ha='center', va='center', 
                                bbox=bbox_props)
            else:
                # If weights don't exist, draw normal edges
                nx.draw_networkx_edges(G, pos, edge_color=self.edge_color_scheme, 
                                    width=settings['edge_width'], alpha=settings['edge_alpha'])
        else:
            # Draw standard edges
            nx.draw_networkx_edges(G, pos, edge_color=self.edge_color_scheme, 
                                width=settings['edge_width'], alpha=settings['edge_alpha'])
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=self.node_color_scheme, node_size=settings['node_size'])
        
        # Only draw labels if font size is not zero
        if settings['font_size'] > 0:
            # For larger graphs, use a smaller font for labels
            nx.draw_networkx_labels(G, pos, font_size=settings['font_size'], 
                                  font_weight='bold', font_family='sans-serif')
        
        # Title with adaptive font size
        title_fontsize = min(16, max(10, settings['font_size'] + 4))
        plt.title("Network of Clients", fontsize=title_fontsize)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'{self.save_path}/{file_name}.png')
        
        return pos
    
    def plot_data_distribution(self, client_data: Dict[int, Dict], num_classes: int = 10,
                              file_name: str = 'data_distribution') -> None:
        """
        Plot heatmap of label distribution across clients with adaptive sizing.
        
        Args:
            client_data (Dict): Dictionary of client data
            num_classes (int): Number of classes in the dataset
            file_name (str): Name for the saved file
        """
        num_clients = len(client_data)
        
        # Prepare data
        label_counts = np.zeros((num_clients, num_classes), dtype=int)
        
        for client_id in range(num_clients):
            targets = torch.tensor(client_data[client_id]['train_dataset'].dataset.targets)
            train_labels = targets[client_data[client_id]['train_indices']].tolist()
           #train_labels = [int(label) for label in client_data[client_id]['train_dataset'].dataset.targets[client_data[client_id]['train_indices']]]
            counts = Counter(train_labels)
            for label, count in counts.items():
                label_counts[client_id, label] = count
        
        # Adaptive figure size based on number of clients and classes
        figsize_width = max(8, min(20, num_classes * 0.8))
        figsize_height = max(6, min(20, num_clients * 0.5))
        
        # Create DataFrame for seaborn
        df = pd.DataFrame(label_counts, 
                         columns=[f'{i}' for i in range(num_classes)], 
                         index=[f'{i}' for i in range(num_clients)])
        
        plt.figure(figsize=(figsize_width, figsize_height))
        
        # Calculate the max count for proper color scaling
        max_count = label_counts.max()
        
        # Adjust annotation size based on the number of cells
        annot_size = max(6, min(10, 300 / (num_clients * num_classes)))
        
        # Determine if annotations should be shown
        show_annot = (num_clients * num_classes) <= 500
        
        # Create heatmap with adaptive settings
        sns.heatmap(df, annot=show_annot, fmt='d', cmap=self.heatmap_cmap, 
                   vmin=0, vmax=max_count, annot_kws={"size": annot_size})
        
        # Adjust title and label sizes based on figure size
        title_size = max(12, min(16, figsize_width * 0.8))
        label_size = max(10, min(14, figsize_width * 0.6))
        
        plt.title('Label Distribution Across Clients', fontsize=title_size)
        plt.xlabel('Class ID', fontsize=label_size)
        plt.ylabel('Client ID', fontsize=label_size)
        
        # Adjust tick label sizes
        tick_size = max(6, min(10, 300 / max(num_clients, num_classes)))
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/{file_name}.png')
    
    def plot_selection_probability(self, P: np.ndarray, file_name: str = 'selection_probability_matrix') -> None:
        """
        Plot transition probability matrix with adaptive sizing.
        
        Args:
            P (np.ndarray): Transition probability matrix
            file_name (str): Name for the saved file
        """
        num_clients = P.shape[0]
        
        # Adaptive figure size based on matrix dimensions
        figsize_base = max(5, min(20, num_clients * 0.5))
        figsize = (figsize_base, figsize_base * 0.8)
        
        plt.figure(figsize=figsize)
        
        # Mask zero probabilities
        mask = P == 0
        
        # Adjust annotation size based on matrix size
        annot_size = max(6, min(10, 200 / num_clients))
        
        # Determine if annotations should be shown
        show_annot = num_clients <= 50
        
        # Create heatmap with adaptive settings
        sns.heatmap(P, annot=show_annot, cmap=self.transition_cmap, fmt=".2f", 
                   cbar=num_clients <= 50, mask=mask, 
                   annot_kws={"size": annot_size})
        
        # Adjust title and label sizes based on figure size
        title_size = max(10, min(14, figsize_base * 0.6))
        label_size = max(8, min(12, figsize_base * 0.5))
        
        plt.title("Transition Probability Matrix", fontsize=title_size)
        plt.xlabel("To Client", fontsize=label_size)
        plt.ylabel("From Client", fontsize=label_size)
        
        # Adjust tick label sizes
        tick_size = max(6, min(9, 150 / num_clients))
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/{file_name}.png')
        
    def plot_transition_graph(self, P: np.ndarray, pos: Optional[Dict] = None, 
                             file_name: str = 'selection_graph') -> None:
        """
        Creates and plots a directed graph from a transition probability matrix with adaptive settings.
        
        Args:
            P (np.ndarray): Transition probability matrix
            pos (Dict, optional): Node positions for the graph
            file_name (str): Name for the saved file
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
        
        # Get adaptive settings
        settings = self._calculate_adaptive_settings(G)
        
        # If positions not provided, generate them
        if pos is None:
            pos = self._get_best_layout(G, 'auto')
        
        # Extract edge weights for coloring
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        
        # Normalize weights for colormap
        if not edge_weights or max(edge_weights) == min(edge_weights):
            norm = mcolors.Normalize(vmin=0, vmax=1)
        else:
            norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
            
        # Map weights to colors
        edge_colors = [self.transition_cmap(norm(weight)) for weight in edge_weights]
        
        # Create the plot with adaptive size
        plt.figure(figsize=settings['figure_size'])
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=settings['node_size'], node_color=self.node_color_scheme)
        
        # Adaptive connection style based on graph size
        if num_nodes <= 20:
            conn_style = 'arc3,rad=0.1'
        elif num_nodes <= 50:
            conn_style = 'arc3,rad=0.08'
        else:
            conn_style = 'arc3,rad=0.05'
            
        # Adaptive arrow size
        arrow_size = max(10, min(20, settings['node_size'] / 50))
        
        # Draw edges with adaptive settings
        nx.draw_networkx_edges(G, pos,
                            edge_color=edge_colors,
                            width=settings['edge_width'],
                            alpha=settings['edge_alpha'],
                            arrows=True,
                            arrowstyle='->',
                            arrowsize=arrow_size,
                            connectionstyle=conn_style)
        
        # Only draw labels if font size is not zero
        if settings['font_size'] > 0:
            nx.draw_networkx_labels(G, pos, font_size=settings['font_size'], 
                                  font_family='sans-serif')
        
        # Create a colorbar if there are enough edges to warrant it
        if len(edge_weights) > 1:
            sm = cm.ScalarMappable(norm=norm, cmap=self.transition_cmap)
            sm.set_array(edge_weights)
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('Selection Probability', fontsize=max(8, settings['font_size']))
            cbar.ax.tick_params(labelsize=max(6, settings['font_size']-2))
        
        # Title with adaptive font size
        title_fontsize = min(16, max(10, settings['font_size'] + 4))
        plt.title('Selection Probability Graph', fontsize=title_fontsize)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/{file_name}.png')

    def set_color_scheme(self, node_color: str = 'lightblue', edge_color: str = 'gray',
                        edge_cmap: str = 'coolwarm', heatmap_cmap: str = 'YlGnBu',
                        transition_cmap: str = 'viridis') -> None:
        """
        Set custom color schemes for visualizations.
        
        Args:
            node_color (str): Color for nodes
            edge_color (str): Color for edges
            edge_cmap (str): Colormap for weighted edges
            heatmap_cmap (str): Colormap for heatmaps
            transition_cmap (str): Colormap for transition graphs
        """
        self.node_color_scheme = node_color
        self.edge_color_scheme = edge_color
        self.edge_cmap = plt.get_cmap(edge_cmap)
        self.heatmap_cmap = heatmap_cmap
        self.transition_cmap = plt.get_cmap(transition_cmap)
        
# For backward compatibility, create wrapper functions
def plot_topology(G: nx.Graph, layout_type: str = 'auto', Adj: np.ndarray = None, 
                 pos: Optional[Dict] = None, save_folder: str = './', file_name: str = 'original_topology') -> Dict:
    """Plot a network topology using adaptive settings"""
    visualizer = AdaptiveVisualizer(save_folder)
    return visualizer.plot_topology(G, layout_type, Adj, pos, file_name)
    
def plot_data_distribution(client_data: Dict[int, Dict], save_folder: str = './', 
                         file_name: str = 'data_distribution', num_classes: int = 10) -> None:
    """Plot heatmap of label distribution across clients using adaptive settings"""
    visualizer = AdaptiveVisualizer(save_folder)
    visualizer.plot_data_distribution(client_data, num_classes, file_name)
    
def plot_selection_probability(P: np.ndarray, save_folder: str = './', 
                             file_name: str = 'selection_probability_matrix') -> None:
    """Plot transition probability matrix using adaptive settings"""
    visualizer = AdaptiveVisualizer(save_folder)
    visualizer.plot_selection_probability(P, file_name)

def plot_transition_graph(P: np.ndarray, pos: Dict, save_folder: str = './', 
                        file_name: str = 'selection_graph') -> None:
    """Plot transition graph using adaptive settings"""
    visualizer = AdaptiveVisualizer(save_folder)
    visualizer.plot_transition_graph(P, pos, file_name)
