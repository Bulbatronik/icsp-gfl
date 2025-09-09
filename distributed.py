import numpy as np
import wandb


def run_decentralized_fl(clients, rounds, rounds_patience):
    """Main decentralized FL simulation with 3-phase communication"""
    print(f"Selection: {clients[0].selection_method} ({clients[0].selection_ratio*100:.0f}% of neighbors)")
    
    # Check if all clients have the same device and print it
    if hasattr(clients[0], 'device'):
        device = clients[0].device
        all_same_device = all(client.device == device for client in clients.values())
        
        if all_same_device:
            print(f"Running on device: {device}")
        else:
            print("\033[93mWarning: Not all clients are using the same device:\033[0m")
            device_counts = {}
            for client_id, client in clients.items():
                if client.device not in device_counts:
                    device_counts[client.device] = []
                device_counts[client.device].append(client_id)
            
            for device, client_ids in device_counts.items():
                print(f"  - Device {device}: Clients {client_ids}")
    
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
        
        # Log metrics to wandb
        metrics = {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_acc,
        }
        
        # Log individual client metrics
        for client_id, loss in enumerate(train_losses):
            metrics[f'client_{client_id}_loss'] = loss
        
        for client_id, acc in enumerate(test_accuracies):
            metrics[f'client_{client_id}_accuracy'] = acc
            
        wandb.log(metrics, step=round_num+1)
        
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