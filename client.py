from torch import nn, optim
import torch.nn.functional as F
import torch
import random
import networkx as nx
import numpy as np
from scipy.linalg import eigh  # for eigen-decomposition
from scipy.sparse import csgraph
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class DecentralizedClient:
    """Minimal FL client for decentralized learning"""
    def __init__(
            self, client_id, graph, model, 
            train_loader, test_loader, 
            selection_method, num_eig, t, selection_ratio, dist,
            optimizer, lr, rho):
        
        self.client_id = client_id
        self.graph = graph
        self.neighbors = list(self.graph.neighbors(client_id))  # Neighbors of this client
        self.selection_method = selection_method
        self.num_eig = num_eig
        self.selection_ratio = selection_ratio
        self.dist = dist
        self.t = t # diffusion time, small = local, large = global
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.model = model # TODO: Model selection
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.rho = rho
        
        # Graph-based neighbors selection if 'selection_method' != random (or broadcast)
        A = nx.adjacency_matrix(self.graph).toarray()
        #self.similarity_matrix = A
        
        # 1. Heat kernel
        if selection_method == 'heatkernel':
            # TODO: Check if correct
            L = csgraph.laplacian(A, normed=False)  # unnormalized Laplacian

            # Compute first k eigenvectors/eigenvalues
            eigvals, eigvecs = eigh(L)  # L is symmetric, returns sorted eigenvalues

            # Keep first k eigenvectors (excluding the zero eigenvalue if desired)
            eigvals = eigvals[1:num_eig+1] # skip first trivial eigenvalue 0
            eigvecs = eigvecs[:, 1:num_eig+1]

            # Heat/diffusion kernel
            heat_kernel = eigvecs @ np.diag(np.exp(-self.t * eigvals)) @ eigvecs.T

            # Convert kernel to similarity matrix
            similarity_matrix = heat_kernel

        elif selection_method == 'spectr':
            # TODO: Check if correct
            L = csgraph.laplacian(A, normed=True)
            
            # Compute first k eigenvectors/eigenvalues
            eigvals, eigvecs = eigh(L)  # L is symmetric, returns sorted eigenvalues
            # Normalize the eigenvectors
            eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=1, keepdims=True)
            
            # Keep first k eigenvectors (excluding the zero eigenvalue if desired)
            node_embeddings = eigvecs[:, 1:num_eig+1]
           
            # Compute similarity matrix based on chosen distance
            if dist == 'cosine':
                similarity_matrix = cosine_similarity(node_embeddings)
            elif dist == 'eucl':
                distances = euclidean_distances(node_embeddings)
                sigma = np.mean(distances)
                similarity_matrix = np.exp(-distances**2 / (2 * sigma**2))
            else:
                raise ValueError(f"Unknown distance metric: {dist}")                     
        else:
            similarity_matrix = A # Default to adjacency if unknown method
            if selection_method == "broadcast":
                self.selection_ratio = 1.0 # Broadcast to all neighbors
            
        self.A_tilde = A * similarity_matrix
        
        # Store the similarity between the nodes
        self.neighbors_sim = [similarity_matrix[client_id, nbr] for nbr in self.neighbors]
        # Compute probabilities (similarities can be negative)
        
        #self.neighbors_proba = self.neighbors_sim / np.sum(self.neighbors_sim) # TODO: MB Softmax
        self.neighbors_proba = np.exp(self.neighbors_sim) / np.sum(np.exp(self.neighbors_sim))
        
        
        #if self.client_id == 0: # Plot the graph, but this time with the weights
        #   ...  # TODO 
        
        
    def train_local(self, epochs=1):
        """Train locally on client data"""
        self.model.train()
        total_loss = 0
        for _ in range(epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        """Evaluate on test data"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return 100. * correct / total
    
    def get_parameters(self):
        """Get model parameters"""
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, params):
        """Set model parameters"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(params[name])
    
    def select_neighbors(self):
        """Select subset of neighbors for communication"""
        #if not self.neighbors:
        #    return []
        
        # Based on the similarity with your neighbors do the selection
        return np.random.choice(self.neighbors, int(len(self.neighbors) * self.selection_ratio), 
                         replace=False, p=self.neighbors_proba)
        
    
    
    def transmit_to_selected(self, selected_neighbors, all_clients):
        """Transmit model parameters to selected neighbors"""
        my_params = self.get_parameters()
        transmitted_to = []
        
        for neighbor_id in selected_neighbors:
            if neighbor_id in all_clients:
                # Store received model in neighbor's received_models
                all_clients[neighbor_id].receive_model(self.client_id, my_params)
                transmitted_to.append(neighbor_id)
        
        return transmitted_to
    
    def receive_model(self, sender_id, model_params):
        """Receive and store model parameters from another client"""
        if not hasattr(self, 'received_models'):
            self.received_models = {}
        self.received_models[sender_id] = model_params
    
    def aggregate_received_models(self):
        """Aggregate own model with received models from other clients"""
        current_params = self.get_parameters()
        
        if not hasattr(self, 'received_models') or not self.received_models:
            return  # No models received, keep current model
        
        # Get all received model parameters
        received_params_list = list(self.received_models.values())
        
        # Average all received parameters
        avg_received_params = {}
        for name in current_params.keys():
            received_tensors = [rp[name] for rp in received_params_list if name in rp]
            if received_tensors:
                avg_received_params[name] = torch.stack(received_tensors).mean(dim=0)
            else:
                avg_received_params[name] = current_params[name]
        
        # Apply aggregation: (1-rho)*own + rho*received_avg
        aggregated_params = {}
        for name in current_params.keys():
            aggregated_params[name] = (1 - self.rho) * current_params[name] + self.rho * avg_received_params[name]
        
        self.set_parameters(aggregated_params)
        
        # Clear received models for next round
        self.received_models = {}
        
        
class SimpleMNISTModel(nn.Module):
    """Lightweight CNN for MNIST"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)