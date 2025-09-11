import numpy as np
import wandb
import torch
import multiprocessing as mp
from tqdm import tqdm
import time
import os


def train_client_process(client_id, client_data, device=None):
    """Function to train a client in a separate process"""
    # If we're using CUDA, set the device
    if device and device.startswith('cuda'):
        # Extract device index if specified
        device_idx = 0
        if ':' in device:
            device_idx = int(device.split(':')[1])
        torch.cuda.set_device(device_idx)
    
    # Recreate client in this process
    client = client_data['client']
    
    # Train the client
    loss = client.train_local()
    
    # Return the updated model parameters
    return client_id, loss, client.get_parameters()


def evaluate_client_process(client_id, client_data, device=None):
    """Function to evaluate a client in a separate process"""
    # If we're using CUDA, set the device
    if device and device.startswith('cuda'):
        # Extract device index if specified
        device_idx = 0
        if ':' in device:
            device_idx = int(device.split(':')[1])
        torch.cuda.set_device(device_idx)
    
    # Recreate client in this process
    client = client_data['client']
    
    # Evaluate the client
    acc = client.evaluate()
    return client_id, acc


def run_decentralized_fl(clients, rounds, rounds_patience, num_parallel=None, device=None):
    """Main decentralized FL simulation with 3-phase communication using multiprocessing"""
    print(f"Selection: {clients[0].selection_method} ({clients[0].selection_ratio*100:.0f}% of neighbors)")
    
    # Determine if we're using CUDA
    using_cuda = device == 'cuda' and torch.cuda.is_available()
    
    # Set up multiprocessing
    if not mp.get_start_method(allow_none=True):
        mp.set_start_method('spawn')
    
    # Set default num_parallel if not provided
    if num_parallel is None:
        if using_cuda:
            # Default to number of CUDA devices
            num_parallel = torch.cuda.device_count()
        else:
            # Default to CPU cores
            num_parallel = max(1, os.cpu_count() - 1)
    
    print(f"Training with up to {num_parallel} clients in parallel using multiprocessing")
    
    # Training rounds
    results = []
    patience = 0
    best_test_acc = float('-inf')
    
    for round_num in range(rounds):
        round_start_time = time.time()
        
        # Patience for early stopping
        if patience >= rounds_patience:
            print(f"Early stopping at round {round_num} due to no improvement in accuracy.")
            break
        
        # Phase 1: Each client trains locally in parallel
        train_losses = [0.0] * len(clients)
        
        # Create a pool of worker processes for training
        try:
            pool = mp.Pool(processes=num_parallel)
            
            # Submit training tasks
            print(f"Round {round_num+1}: Training clients in parallel...")
            training_results = []
            
            # Create simplified client data for each client to pass to worker processes
            client_data_dict = {}
            for client_id, client in clients.items():
                client_data_dict[client_id] = {'client': client}
            
            for client_id, client_data in client_data_dict.items():
                # Use device string instead of queue
                device_str = str(clients[client_id].device)
                result = pool.apply_async(train_client_process, args=(client_id, client_data, device_str))
                training_results.append(result)
            
            # Get results and update client parameters
            for result in tqdm(training_results, desc="Training"):
                client_id, loss, params = result.get()
                train_losses[client_id] = loss
                # Update client parameters with the trained ones
                clients[client_id].set_parameters(params)
            
            # Close the pool
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print("\nTraining interrupted. Terminating worker processes...")
            pool.terminate()
            pool.join()
            print("Worker processes terminated.")
            raise
        
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
        
        # Evaluate all clients in parallel
        test_accuracies = [0] * len(clients)
        
        # Create a new pool for evaluation
        try:
            pool = mp.Pool(processes=num_parallel)
            
            # Submit evaluation tasks
            print(f"Round {round_num+1}: Evaluating clients in parallel...")
            eval_results = []
            
            # Update client data
            client_data_dict = {}
            for client_id, client in clients.items():
                client_data_dict[client_id] = {'client': client}
                
            for client_id, client_data in client_data_dict.items():
                # Use device string instead of queue
                device_str = str(clients[client_id].device)
                result = pool.apply_async(evaluate_client_process, args=(client_id, client_data, device_str))
                eval_results.append(result)
            
            # Get evaluation results
            for result in tqdm(eval_results, desc="Evaluating"):
                client_id, acc = result.get()
                test_accuracies[client_id] = acc
            
            # Close the pool
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print("\nEvaluation interrupted. Terminating worker processes...")
            pool.terminate()
            pool.join()
            print("Worker processes terminated.")
            raise
        
        # Round timing
        round_end_time = time.time()
        round_time = round_end_time - round_start_time
        
        # Store round results
        avg_loss = np.mean(train_losses)
        avg_acc = np.mean(test_accuracies)
        results.append({
            'round': round_num + 1, 
            'loss': avg_loss, 
            'accuracy': avg_acc,
            'round_time': round_time
        })
        
        # Log metrics to wandb
        metrics = {}
        # Log individual client metrics
        for client_id, loss in enumerate(train_losses):
            metrics[f'client_{client_id}_loss'] = loss
        for client_id, acc in enumerate(test_accuracies):
            metrics[f'client_{client_id}_accuracy'] = acc
        # Log aggregated metrics
        metrics.update({
            'avg_loss': avg_loss,
            'avg_accuracy': avg_acc,
            'round_time': round_time
        })
        # Log to wandb without specifying step to ensure monotonically increasing steps
        wandb.log(metrics)
        
        print(f"Round {round_num + 1}: Loss={avg_loss:.3f}, Accuracy={avg_acc:.1f}%, Time={round_time:.2f}s")

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