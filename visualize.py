# Import necessary libraries for decentralized federated learning simulation
import torch

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import random
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Visualization Functions for Network Topologies
def plot_topology(G: nx.Graph, title: str, layout_type: str = 'spring') -> None:
    """Plot a network topology using matplotlib"""
    plt.figure(figsize=(10, 8))
    
    # Choose layout
    if layout_type == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'shell':
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Draw the network
    nx.draw(G, pos, 
            with_labels=True, 
            node_color='lightblue', 
            node_size=1000,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            width=2,
            alpha=0.7)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_interactive_topology(G: nx.Graph, title: str) -> None:
    """Create an interactive plot using Plotly"""
    # Get layout positions
    pos = nx.spring_layout(G, seed=42)
    
    # Extract edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=2, color='#888'),
                           hoverinfo='none',
                           mode='lines')
    
    # Extract nodes
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'Client {node}')
        
        # Node info for hover
        adjacencies = list(G.neighbors(node))
        node_info.append(f'Client {node}<br>Connections: {len(adjacencies)}<br>Neighbors: {adjacencies}')
    
    # Create node trace
    node_trace = go.Scatter(x=node_x, y=node_y,
                           mode='markers+text',
                           hoverinfo='text',
                           text=node_text,
                           textposition="middle center",
                           hovertext=node_info,
                           marker=dict(showscale=True,
                                     colorscale='Blues',
                                     reversescale=True,
                                     color=[],
                                     size=30,
                                     colorbar=dict(
                                         thickness=15,
                                         len=0.5,
                                         x=1.02,
                                         xanchor="left",
                                         title="Node Connections"
                                     ),
                                     line=dict(width=2)))
    
    # Color nodes by number of connections
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    
    node_trace.marker.color = node_adjacencies
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                        title=dict(text=title, font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    fig.show()