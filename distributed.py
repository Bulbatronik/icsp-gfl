from partitioner import DataDistributor
from client import DecentralizedClient
import numpy as np
import torch


def run_decentralized_fl(clients, rounds, rounds_patience):
    """Main decentralized FL simulation with 3-phase communication"""
    print(f"Selection: {clients[0].selection_method} ({clients[0].selection_ratio*100:.0f}% of neighbors)")
    
    # Check if all clients have the same device and print it
    if hasattr(clients[0], 'device'):
        device = clients[0].device
        print(f"Running on device: {device}")
    
    # Training rounds
    results = []
    patience = 0
    best_test_acc = float('-inf')
    
    for round_num in range(rounds):
        # Patience for early stopping
        if patience >= rounds_patience:
            print(f"Early stopping at round {round_num} due to no improvement in accuracy.")
            break
        
        # Phase 1: Each client trains locally
        train_losses = []
        for client_id, client in clients.items():
            print(f"Client {client_id} training locally...")
            loss = client.train_local()
            train_losses.append(loss)
        
        # Phase 2: Client selection and model transmission
        transmission_log = {}
        for client_id, client in clients.items():
            # Each client selects neighbors to transmit its model to
            selected_neighbors = client.select_neighbors()
            transmission_log[client_id] = selected_neighbors
            
            # Transmit model to selected neighbors directly
            if len(selected_neighbors) > 0:
                client.transmit_to_selected(selected_neighbors, clients)
        
        # Phase 3: Model aggregation
        for client_id, client in clients.items():
            print(f"Client {client_id} aggregating models...")
            client.aggregate_received_models()
        
        # Evaluate all clients
        test_accuracies = []
        for client_id, client in clients.items():
            print(f"Client {client_id} evaluating...")
            acc = client.evaluate()
            test_accuracies.append(acc)
        
        # Store round results
        avg_loss = np.mean(train_losses)
        avg_acc = np.mean(test_accuracies)
        results.append({'round': round_num + 1, 'loss': avg_loss, 'accuracy': avg_acc})
        
        print(f"Round {round_num + 1}: Loss={avg_loss:.3f}, Accuracy={avg_acc:.1f}%")

        # Early stopping based on accuracy
        if avg_acc > best_test_acc:
            best_test_acc = avg_acc
            patience = 0
        else:
            patience += 1
            if patience >= rounds_patience:
                print(f"Early stopping at round {round_num + 1} due to no improvement in accuracy.")
                break
    return results