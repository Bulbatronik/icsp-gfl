from torch import nn, optim
import torch.nn.functional as F
import torch
import random


class DecentralizedClient:
    """Minimal FL client for decentralized learning"""
    def __init__(
            self, client_id, model, 
            train_loader,test_loader, 
            neighbors, selection_method, selction_ratio, dist,
            optimizer, lr, rho):
        
        self.client_id = client_id
        self.neighbors = neighbors
        self.selection_method = selection_method
        self.selction_ratio = selction_ratio
        self.dist = dist
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.model = model #SimpleMNISTModel()
        # TODO: TRAINING PART (OPTIMIZER, LR)
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.rho = rho
        
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
    
    def select_neighbors(self, selection_strategy='random', selection_ratio=0.5):
        """Select subset of neighbors for communication"""
        if not self.neighbors:
            return []
        
        if selection_strategy == 'random':
            num_selected = max(1, int(len(self.neighbors) * selection_ratio)) # TODO
            return random.sample(self.neighbors, num_selected)
        elif selection_strategy == 'broadcast':
            return self.neighbors.copy()
        #elif "spectrum":
            
        
        else:   
            return []
    
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