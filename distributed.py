from partitioner import DataDistributor
from client import DecentralizedClient
import numpy as np


def run_decentralized_fl(graph, num_clients=6, num_rounds=5, local_epochs=2, rho=0.2, selection_strategy='random', selection_ratio=0.5):
    """Main decentralized FL simulation with 3-phase communication"""
    #print(f"Running {topology_type} topology: {num_clients} clients, {num_rounds} rounds")
    print(f"Selection: {selection_strategy} ({selection_ratio*100:.0f}% of neighbors)")
    # Get neighbor information for each client
    neighbor_info = {i: list(graph.neighbors(i)) for i in range(num_clients)}
    
    # Create data distribution and clients
    data_dist = DataDistributor(num_clients)
    data_dist.distribute_iid_data(batch_size=32)
    
    clients = {}
    for client_id in range(num_clients):
        train_loader = data_dist.get_client_loader(client_id, 'train')
        test_loader = data_dist.get_client_loader(client_id, 'test')
        clients[client_id] = DecentralizedClient(client_id, train_loader, test_loader, neighbor_info[i])
    
    # Training rounds
    results = []
    
    for round_num in range(num_rounds):
        # Phase 1: Each client trains locally
        train_losses = []
        for client_id, client in clients.items():
            loss = client.train_local(epochs=local_epochs)
            train_losses.append(loss)
        
        # Phase 2: Client selection and model transmission
        transmission_log = {}
        for client_id, client in clients.items():
            # Each client selects neighbors to request models from
            selected_neighbors = client.select_neighbors(selection_strategy, selection_ratio)
            transmission_log[client_id] = selected_neighbors
            
            # Request models from selected neighbors (they transmit to this client)
            for neighbor_id in selected_neighbors:
                if neighbor_id in clients:
                    clients[neighbor_id].transmit_to_selected([client_id], clients)
        
        # Phase 3: Model aggregation
        for client_id, client in clients.items():
            client.aggregate_received_models(rho=rho)
        
        # Evaluate all clients
        test_accuracies = []
        for client_id, client in clients.items():
            acc = client.evaluate()
            test_accuracies.append(acc)
        
        # Store round results
        avg_loss = np.mean(train_losses)
        avg_acc = np.mean(test_accuracies)
        results.append({'round': round_num + 1, 'loss': avg_loss, 'accuracy': avg_acc})
        
        print(f"Round {round_num + 1}: Loss={avg_loss:.3f}, Accuracy={avg_acc:.1f}%")
    
    return results