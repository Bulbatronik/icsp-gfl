from torch import nn, optim
import torch
from copy import deepcopy
from math import ceil
import networkx as nx
import numpy as np
from scipy.linalg import eigh  # for eigen-decomposition
from scipy.sparse import csgraph
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm


class DecentralizedClient:
    """Minimal FL client for decentralized learning"""
    def __init__(
            self, client_id, graph, model, 
            train_loader, test_loader, 
            selection_method, num_eig, t, tau, selection_ratio, dist,
            optimizer, epochs, lr, rho, device=None, **kwargs):
        
        self.client_id = client_id
        self.graph = graph
        self.neighbors = list(self.graph.neighbors(client_id))  # Neighbors of this client
        self.selection_method = selection_method
        self.num_eig = num_eig
        self.selection_ratio = selection_ratio
        self.dist = dist
        self.t = t # Diffusion time, small = local, large = global
        self.tau = tau # Temperature for the client selection
        
        # Set device with better error handling
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Client {client_id}: No device specified, using {self.device}")
        else:
            try:
                if isinstance(device, str):
                    self.device = torch.device(device)
                else:
                    self.device = device
                if self.device.type == 'cuda' and not torch.cuda.is_available():
                    print(f"\033[93mWarning: Client {client_id} requested CUDA but it's not available. Falling back to CPU.\033[0m")
                    self.device = torch.device('cpu')
            except Exception as e:
                print(f"\033[91mError: Client {client_id} couldn't use device {device}. Error: {e}. Falling back to CPU.\033[0m")
                self.device = torch.device('cpu')
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.model = model # TODO: Model selections
        # Ensure model is on the correct device
        if hasattr(model, 'device') and model.device != self.device:
            self.model.to(self.device)
            
        self.epochs = epochs
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.rho = rho
        
        # Graph-based neighbors selection if 'selection_method' != random (or broadcast)
        A = nx.adjacency_matrix(self.graph, nodelist=range(graph.number_of_nodes())).toarray()
        #self.similarity_matrix = A
        if self.client_id == 0:
            print(A)
        # 1. Heat kernel
        if selection_method == 'heatkernel':
            L = csgraph.laplacian(A, normed=False)  # unnormalized Laplacian

            # Compute first k eigenvectors/eigenvalues
            eigvals, eigvecs = eigh(L)  # L is symmetric, returns sorted eigenvalues
            
            if self.client_id == 0:
                print(f"Eigenvalues of L: {eigvals} ")
           
            # Keep first k eigenvectors (excluding the zero eigenvalue if desired)
            eigvals = eigvals[1:num_eig+1] # skip first trivial eigenvalue 0
            eigvecs = eigvecs[:, 1:num_eig+1]
            #eigvals = #eigvals[:num_eig]
            #eigvecs = #eigvecs[:, :num_eig]
            
            # Heat/diffusion kernel
            heat_kernel = eigvecs @ np.diag(np.exp(-self.t * eigvals)) @ eigvecs.T

            # OPT1: Convert kernel to similarity matrix
            #similarity_matrix = heat_kernel

            # OPT2: cossim(i, j) = K(i, j) / sqrt(K(i, i) * K(j, j))
            #norms = np.sqrt(np.diag(heat_kernel))
            #similarity_matrix = heat_kernel / np.outer(norms, norms)
            
            # OPT3: diff dist D(i, j)^2 = K(i, i) + K(j, j) - 2K(i, j)
            similarity_matrix = 1/np.maximum(0, np.diag(heat_kernel)[:, None] + np.diag(heat_kernel)[None, :] - 2 * heat_kernel)
            #sigma = np.mean(distances) # Gaussian Kernel
            #similarity_matrix = np.exp(-distances**2 / (2 * sigma**2))
            
            
        elif selection_method == 'spectrclust':
            L = csgraph.laplacian(A, normed=True)
            
            # Compute first k eigenvectors/eigenvalues
            eigvals, eigvecs = eigh(L)  # L is symmetric, returns sorted eigenvalues
            
            if self.client_id == 0:
                print(f"Eigenvalues of L: {eigvals} ")
            
            # Normalize the eigenvectors
            eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=1, keepdims=True)
            
            # Keep first k eigenvectors (excluding the zero eigenvalue if desired)
            node_embeddings = eigvecs[:, 1:num_eig+1]
           
            # Compute similarity matrix based on chosen distance
            if dist == 'cosine':
                similarity_matrix = cosine_similarity(node_embeddings)
            elif dist == 'eucl':
                distances = euclidean_distances(node_embeddings)
                sigma = np.mean(distances) # Gaussian Kernel
                similarity_matrix = np.exp(-distances**2 / (2 * sigma**2))
            else:
                raise ValueError(f"Unknown distance metric: {dist}")                     
        else:
            similarity_matrix = deepcopy(A) # Default to adjacency if unknown method
            if self.client_id == 0:
                print("Sim mtrx")
                print(similarity_matrix)
            
            if selection_method == "broadcast":
                self.selection_ratio = 1.0 # Broadcast to all neighbors
            elif selection_method == "nofed":
                self.selection_ratio = 0.0 # No communication
            elif selection_method == "random":
                self.selection_ratio = selection_ratio
            elif selection_method == "gradients":
                # Gradient-based selection - similarity computed dynamically
                self.selection_ratio = selection_ratio
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")
            
        self.A_tilde = np.multiply(A, similarity_matrix)
        if self.client_id == 0:
            print("A_tilde")
            print(self.A_tilde)
        # Store the similarity between the nodes
        self.neighbors_sim = [similarity_matrix[client_id, nbr] for nbr in self.neighbors]
        
        # Compute probabilities (similarities can be negative)
        # Apply softmax for other methods
        self.neighbors_proba = np.exp(np.array(self.neighbors_sim) / self.tau) / np.sum(np.exp(np.array(self.neighbors_sim) / self.tau))
        
        # For gradient-based selection
        self.gradient_vector = None
        self.params_before_training = None
        self.received_gradients = {}
        #if selection_method in ["random", "nofed", "broadcast"] or (len(set(self.neighbors_sim)) == 1 and self.neighbors):
        #    # Use uniform distribution for random methods or when all similarities are equal
        #    self.neighbors_proba = np.ones(len(self.neighbors)) / len(self.neighbors) if self.neighbors else np.array([])
        #else:
        #    # Apply softmax for other methods
        #    self.neighbors_proba = np.exp(np.array(self.neighbors_sim) / self.tau) / np.sum(np.exp(np.array(self.neighbors_sim) / self.tau))
        
    def train_local(self):
        """Train locally on client data"""
        # Store parameters before training for gradient computation
        if self.selection_method == 'gradients':
            self.params_before_training = self.get_parameters()
        
        self.model.train()
        
        epochs_iter = tqdm(range(self.epochs))
        for _ in epochs_iter:
            total_loss = 0
            for data, target in self.train_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            epoch_loss = total_loss / len(self.train_loader)
            # Update tqdm with current loss
            epochs_iter.set_postfix(loss=f"{epoch_loss:.4f}")
        
        # Compute gradient vector for gradient-based selection
        if self.selection_method == 'gradients':
            self._compute_gradient_vector()
        
        return epoch_loss  # Return only the last epoch's loss
    
    def evaluate(self):
        """Evaluate on test data"""
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                total_loss += loss.item()
        return 100. * correct / total, total_loss / len(self.test_loader)
    
    def get_parameters(self):
        """Get model parameters"""
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, params):
        """Set model parameters"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    # Make sure the parameter is on the same device
                    param.copy_(params[name].to(self.device))
    
    def select_neighbors(self):
        """Select subset of neighbors for communication"""
        if not self.neighbors:
            return []
        
        # For gradient-based selection, compute similarities dynamically
        if self.selection_method == 'gradients':
            if self.gradient_vector is not None and self.received_gradients:
                # Compute gradient similarities with neighbors who shared gradients
                grad_sims = []
                valid_neighbors = []
                
                for nbr in self.neighbors:
                    if nbr in self.received_gradients:
                        # Compute cosine similarity between gradient vectors
                        sim = self._cosine_similarity(self.gradient_vector, self.received_gradients[nbr])
                        grad_sims.append(sim)
                        valid_neighbors.append(nbr)
                
                if valid_neighbors:
                    # Use softmax with temperature for selection probabilities
                    grad_probs = np.exp(np.array(grad_sims) / self.tau)
                    grad_probs = grad_probs / grad_probs.sum()
                    
                    num_select = max(1, ceil(len(valid_neighbors) * self.selection_ratio))
                    print(f"Client {self.client_id} selecting neighbours {valid_neighbors} with probs {grad_probs}")
                    return np.random.choice(valid_neighbors, num_select, replace=False, p=grad_probs)
            
            # Fallback to uniform random if no gradient info available
            num_select = max(1, ceil(len(self.neighbors) * self.selection_ratio))
            print(f"Client {self.client_id} selecting neighbours {valid_neighbors} uniformly")
            return np.random.choice(self.neighbors, num_select, replace=False)
        
        # Based on the similarity with your neighbors do the selection
        num_select = max(1, ceil(len(self.neighbors) * self.selection_ratio))
        print(f"Client {self.client_id} selecting neighbours {self.neighbors} with probs {self.neighbors_proba}")
        return np.random.choice(self.neighbors, num_select, replace=False, p=self.neighbors_proba)
        
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
    
    def share_gradient_with_neighbors(self, all_clients):
        """Share gradient vector with all neighbors (lightweight exchange)"""
        if self.selection_method == 'gradients' and self.gradient_vector is not None:
            for neighbor_id in self.neighbors:
                if neighbor_id in all_clients:
                    all_clients[neighbor_id].receive_gradient(self.client_id, self.gradient_vector)
    
    def receive_gradient(self, sender_id, gradient_vector):
        """Receive gradient vector from a neighbor"""
        self.received_gradients[sender_id] = gradient_vector
    
    def _compute_gradient_vector(self):
        """Compute lightweight gradient vector as parameter update direction"""
        if self.params_before_training is None:
            return
        
        params_after = self.get_parameters()
        
        # Compute update: params_after - params_before (gradient direction)
        grad_list = []
        for name in params_after.keys():
            update = params_after[name] - self.params_before_training[name]
            grad_list.append(update.cpu().flatten())
        
        # Flatten to single vector
        gradient_flat = torch.cat(grad_list).numpy()
        
        # Normalize for memory efficiency and to focus on direction
        norm = np.linalg.norm(gradient_flat)
        if norm > 1e-10:
            self.gradient_vector = gradient_flat / norm
        else:
            self.gradient_vector = gradient_flat
    
    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors (lightweight)"""
        # Vectors are already normalized in _compute_gradient_vector
        return np.dot(vec1, vec2)
    
    def aggregate_received_models(self):
        """Aggregate own model with received models from other clients"""
        current_params = self.get_parameters()
        
        if not hasattr(self, 'received_models') or not self.received_models:
            print("Keeping current model as the new one.")
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
        
def constr_prob_matrix(clients):
    """Recover the transition probability matrix"""
    n = len(clients)
    P = np.zeros((n, n))
    for client_id, client in clients.items():
        for nbr, proba in zip(client.neighbors, client.neighbors_proba):
            P[client_id, nbr] = proba
    # Check that rows sum to 1
    assert np.allclose(P.sum(axis=1), 1), "Rows of probability matrix do not sum to 1"
    return P
    
