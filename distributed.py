import numpy as np
import wandb


def run_decentralized_fl(clients, rounds, rounds_patience_acc, rounds_patience_loss):
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
    patience_acc = 0
    patience_loss = 0
    best_test_acc = float('-inf')
    best_test_loss = float('+inf')
    
    for round_num in range(rounds):
        # No early stopping check here - we'll do it after evaluating
        
        # Phase 1: Each client trains locally
        train_losses = []
        for client_id, client in clients.items():
            print(f"Client {client_id} training locally...")
            loss = client.train_local()
            train_losses.append(loss)
        
        # Phase 1.5: Gradient exchange (lightweight, only for gradient-based selection)
        if clients[0].selection_method == 'gradients':
            for client_id, client in clients.items():
                client.share_gradient_with_neighbors(clients)
        
        # Phase 2: Client selection and model transmission
        transmission_log = {}
        for client_id, client in clients.items():
            # Each client selects neighbors to transmit its model to
            selected_neighbors = client.select_neighbors()
            print(f"Client {client_id} selected neighbors: {selected_neighbors} from {client.neighbors}")
            transmission_log[client_id] = selected_neighbors
            
            # Transmit model to selected neighbors directly
            if len(selected_neighbors) > 0:
                client.transmit_to_selected(selected_neighbors, clients)
        
        # Phase 3: Model aggregation
        for client_id, client in clients.items():
            print(f"Client {client_id} aggregating models...")
            client.aggregate_received_models()
        
        # Evaluate all clients
        test_accuracies, test_losses = [], []
        for client_id, client in clients.items():
            print(f"Client {client_id} evaluating...")
            acc, loss = client.evaluate()
            test_accuracies.append(acc)
            test_losses.append(loss)    
               
        # Store round results
        avg_train_loss = np.mean(train_losses)
        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accuracies)
        
        results.append({'round': round_num + 1, 'train_loss': avg_train_loss, 'test_loss': avg_test_loss, 'test_accuracy': avg_test_acc})
        
        # Log metrics to wandb
        metrics = {}
        # Log individual client metrics
        for client_id, loss in enumerate(train_losses):
            metrics[f'client_{client_id}_loss'] = loss
        for client_id, acc in enumerate(test_accuracies):
            metrics[f'client_{client_id}_accuracy'] = acc
        # Log aggregated metrics
        metrics['avg_train_loss'] = avg_train_loss
        metrics['avg_test_accuracy'] = avg_test_acc
        metrics['avg_test_loss'] = avg_test_loss
        metrics['min_train_loss'] = max(train_losses)
        metrics['min_test_loss'] = min(test_losses)
        metrics['max_test_accuracy'] = max(test_accuracies)
        
        

        print(f"Round {round_num + 1}/{rounds}: Train Loss={avg_train_loss:.3f}, Test Loss={avg_test_loss:.3f}, Test Accuracy={avg_test_acc:.1f}%")

        # Early stopping based on accuracy
        #if avg_test_acc > best_test_acc:
        #    best_test_acc = avg_test_acc
        #    patience_acc = 0
        #else:
        #    patience_acc += 1
        
        # Early stopping based on loss
        #if avg_test_loss < best_test_loss:
        #    best_test_loss = avg_test_loss
        #    patience_loss = 0
        #else:
        #    patience_loss += 1
        
        # Store best metrics
        #metrics['best_test_loss'] = best_test_loss
        #metrics['best_test_accuracy'] = best_test_acc
        
        # Log to wandb without specifying step to ensure monotonically increasing steps
        wandb.log(metrics)
        
        #if patience_acc >= rounds_patience_acc:
        #   print(f"Early stopping at round {round_num + 1} due to no improvement in accuracy for {rounds_patience_acc} rounds.")
        #   break
        #elif if patience_loss >= rounds_patience_loss:
        #   print(f"Early stopping at round {round_num + 1} due to no improvement in loss for {rounds_patience_loss} rounds.")
        #   break
        
        
    return results