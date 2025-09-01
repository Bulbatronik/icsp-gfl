import networkx as nx
import numpy as np
from typing import Dict

# Network Topology Creation for Decentralized Federated Learning
class NetworkTopology:
    """Class to create and manage different network topologies for decentralized FL
        Ring - Sequential neighbor connections
        Fully Connected - All clients connected to all others
        Star - One central client connected to all others
        Random - Probabilistic connections with guaranteed connectivity
        Small World - Watts-Strogatz model with good clustering and short paths
        Grid - 2D lattice structure for geographic scenarios
    """
    
    def __init__(self, num_clients: int = 5, topology: str = "random", edge_proba: float = 0.3,
                 k: int = 4, p: float = 0.3):
        self.num_clients = num_clients # min(num_clients, 10)  # Limit to 10 clients
        self.clients = list(range(self.num_clients))
        self.edge_proba = edge_proba
        self.k = k # Each node is connected to k nearest neighbors in the ring topology.
        self.p = p # Probability of rewiring each edge.
        self.topology = topology
        self.G = None
    
    def create_topology(self) -> nx.Graph:
        """Create the specified topology"""
        if self.topology == "ring":
            self.G = self._create_ring_topology()
        elif self.topology == "fullycon":
            self.G = self._create_fully_connected_topology()
        elif self.topology == "star":
            self.G = self._create_star_topology()
        elif self.topology == "random":
            self.G = self._create_random_topology()
        elif self.topology == "smallwrld":
            self.G = self._create_small_world_topology()
        elif self.topology == "grid":
            self.G = self._create_grid_topology()
        elif self.topology == "women":
            self.G = self._women_social_network()
        elif self.topology == "karate":
            self.G = self._karate_graph()
        elif self.topology == "florentine":
            self.G = self._florentine_families()

    def _relabel_nodes(self, G: nx.Graph) -> nx.Graph:
        """Relabel nodes to be sequential integers"""
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        return G
    
    def _create_ring_topology(self) -> nx.Graph:
        """Create a ring topology where each client connects to its neighbors"""
        G = nx.cycle_graph(self.num_clients)
        nx.set_node_attributes(G, {i: f"Client_{i}" for i in range(self.num_clients)}, 'name')
        return G
    
    def _create_fully_connected_topology(self) -> nx.Graph:
        """Create a fully connected topology where every client connects to every other"""
        G = nx.complete_graph(self.num_clients)
        nx.set_node_attributes(G, {i: f"Client_{i}" for i in range(self.num_clients)}, 'name')
        return G
    
    def _create_star_topology(self) -> nx.Graph:
        """Create a star topology with one central client connected to all others"""
        G = nx.star_graph(self.num_clients - 1)
        nx.set_node_attributes(G, {i: f"Client_{i}" for i in range(self.num_clients)}, 'name')
        return G
    
    def _create_random_topology(self) -> nx.Graph:
        """Create a random topology with given edge probability"""
        G = nx.erdos_renyi_graph(self.num_clients, self.edge_proba)
        # Ensure connectivity
        if not nx.is_connected(G):
            # Add edges to make it connected
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                G.add_edge(node1, node2)
        
        nx.set_node_attributes(G, {i: f"Client_{i}" for i in range(self.num_clients)}, 'name')
        return G
    
    def _create_small_world_topology(self) -> nx.Graph:
        """Create a small-world topology using the Watts-Strogatz model."""
        # Ensure k is even and less than num_clients
        k = min(self.k, self.num_clients - 1)
        if k % 2 == 1:
            k -= 1
        if k < 2:
            k = 2
            
        G = nx.watts_strogatz_graph(self.num_clients, k, self.p)
        nx.set_node_attributes(G, {i: f"Client_{i}" for i in range(self.num_clients)}, 'name')
        return G
    
    def _create_grid_topology(self) -> nx.Graph:
        """Create a 2D grid topology (works best with square numbers of clients)"""
        # Find closest square grid
        grid_size = int(np.sqrt(self.num_clients))
        if grid_size * grid_size < self.num_clients:
            grid_size += 1
        
        G = nx.grid_2d_graph(grid_size, grid_size)
        # Remove extra nodes if needed
        nodes_to_remove = []
        for i, node in enumerate(G.nodes()):
            if i >= self.num_clients:
                nodes_to_remove.append(node)
        G.remove_nodes_from(nodes_to_remove)
        G = self._relabel_nodes(G)
        nx.set_node_attributes(G, {i: f"Client_{i}" for i in range(G.number_of_nodes())}, 'name')
        return G
    
    def _women_social_network(self) -> nx.Graph:
        """Davis Southern women social network"""
        G = nx.davis_southern_women_graph()
        G = self._relabel_nodes(G)
        return G
    
    def _karate_graph(self) -> nx.Graph:
        """Zachary’s Karate Club graph"""
        G = nx.karate_club_graph()
        G = self._relabel_nodes(G)
        return G
    
    def _florentine_families(self) -> nx.Graph:
        """Florentine families marriage network"""
        G = nx.florentine_families_graph()
        G = self._relabel_nodes(G)
        return G
    
    def get_topology_info(self) -> Dict:
        """Get information about the topology"""
        return {
            'topology': self.topology,
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'average_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
            'min_degree': min(dict(self.G.degree()).values()),
            'max_degree': max(dict(self.G.degree()).values()),
            'most_frequent_degree': max(set(dict(self.G.degree()).values()), key=list(dict(self.G.degree()).values()).count),
            'diameter': nx.diameter(self.G) if nx.is_connected(self.G) else 'Not connected',
            'clustering_coefficient': nx.average_clustering(self.G),
            'is_connected': nx.is_connected(self.G)
        }

