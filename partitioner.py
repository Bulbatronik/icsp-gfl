import re

import requests


import torch
import numpy as np
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Dict, List


# Data Distribution for Decentralized Federated Learning
class DataDistributor:
    """Class to handle data distribution among federated learning clients"""
    
    def __init__(self, num_clients: int, name: str = 'mnist', partition: str = 'iid',
                 batch_size: int = 32, alpha: float = 0.5, verbose: bool = True):
        self.num_clients = num_clients
        self.name = name.lower()
        self.partition = partition.lower()
        self.batch_size = batch_size
        self.alpha = alpha  # Dirichlet concentration parameter
        self.client_data = {}
        self.client_loaders = {}
        self.verbose = verbose
        
        # for shakespeare
        self.vocab_size = 0      # Store vocab size for model init
        self.char_to_int = {}    # Store mapping
        self.int_to_char = {}    # Store reverse mapping
        
    def load_and_distribute(self):
        # Load the dataset 
        if self.name == 'mnist':
            train_dataset, test_dataset = self._load_mnist_data()
        elif self.name == 'cifar10':
            train_dataset, test_dataset = self._load_cifar10_data()
        elif self.name == 'fmnist':
            train_dataset, test_dataset = self._load_fashion_mnist_data()
        elif self.name == 'shakespeare':
            print('Loading an partitioning shakespeare (NIID only)')
            self._load_and_distribute_shakespeare(self)
        else:
            raise ValueError(f"Unsupported dataset: {self.name}. Supported: 'mnist', 'cifar10'")
        
        if self.name in ["mnist", "cifar10", "fmnist"]:
            # Partition the data among the clients
            if self.partition == 'iid':
                self._distribute_iid_data(train_dataset, test_dataset)
            elif self.partition == 'dir':
                self._distribute_dirichlet_data(train_dataset, test_dataset)
            else:
                raise ValueError(f"Unsupported partitioning method: {self.partition}. Supported: 'iid', 'dir'")    
    
    def _load_and_distribute_shakespeare(self):
        """Downloads, parses, and distributes Shakespeare data by Role"""
        DATA_URL = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
        
        if self.verbose:
            print(f"Downloading Shakespeare dataset from {DATA_URL}...")
        
        try:
            r = requests.get(DATA_URL)
            text = r.text
        except Exception:
            raise ConnectionError("Failed to download Shakespeare dataset")

        # Skip license header
        start_idx = text.find("1\n  From fairest creatures we desire increase,")
        if start_idx == -1: start_idx = 0
        text = text[start_idx:]

        # Parse Roles (Clients)
        pattern = re.compile(r'\n  ([A-Z]+[A-Z\s]*)\.')
        raw_roles = defaultdict(str)
        current_role = None
        last_pos = 0

        for match in pattern.finditer(text):
            if current_role:
                line = text[last_pos:match.start()].replace('\n', ' ').strip()
                raw_roles[current_role] += " " + line
            current_role = match.group(1).strip()
            last_pos = match.end()

        # Select Top N Clients
        sorted_roles = sorted(raw_roles.items(), key=lambda x: len(x[1]), reverse=True)
        top_clients = sorted_roles[:self.num_clients]

        # Build Vocabulary
        subset_text = "".join([c[1] for c in top_clients])
        chars = sorted(list(set(subset_text)))
        self.char_to_int = {ch: i for i, ch in enumerate(chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        if self.verbose:
            print(f"Vocab Size: {self.vocab_size}")
            print(f"Selected {len(top_clients)} clients (Roles)")

        # Distribute to Clients
        for client_id, (role, content) in enumerate(top_clients):
            # Encode content
            encoded = [self.char_to_int[c] for c in content if c in self.char_to_int]
            encoded = np.array(encoded)
            
            # Create Train/Test Split (80/20 as per paper)
            split_idx = int(len(encoded) * 0.8)
            train_data = encoded[:split_idx]
            test_data = encoded[split_idx:]
            
            # Create PyTorch Datasets
            # We wrap in a list because our custom dataset expects a list of arrays (usually for multiple sequences)
            # Here, each client has one long sequence.
            client_train_dataset = ShakespeareDataset([train_data], seq_len=self.seq_len)
            client_test_dataset = ShakespeareDataset([test_data], seq_len=self.seq_len)
            
            # Create Loaders
            train_loader = DataLoader(client_train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(client_test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Store Info
            self.client_data[client_id] = {
                'role': role,
                'train_dataset': client_train_dataset,
                'test_dataset': client_test_dataset,
                'train_samples': len(client_train_dataset),
                'test_samples': len(client_test_dataset)
            }
            
            self.client_loaders[client_id] = {
                'train_loader': train_loader,
                'test_loader': test_loader
            }
            
            if self.verbose:
                print(f"Client {client_id} ({role}): {len(client_train_dataset)} train samples")
                
    def _load_mnist_data(self, data_dir: str = './data') -> Tuple[Dataset, Dataset]:
        """Load MNIST dataset"""
        # Data preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
        
        # Download and load MNIST
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        return train_dataset, test_dataset


    def _load_cifar10_data(self, data_dir: str = './data') -> Tuple[Dataset, Dataset]:
        """Load CIFAR-10 dataset"""
        # Data preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
        ])
        
        # Download and load CIFAR-10
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        return train_dataset, test_dataset
    
    def _load_fashion_mnist_data(self, data_dir: str = './data') -> Tuple[Dataset, Dataset]:
        """Load Fashion-MNIST dataset"""
        # Data preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST normalization
        ])
        
        # Download and load Fashion-MNIST
        train_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        return train_dataset, test_dataset
    
    def _distribute_iid_data(self, train_dataset, test_dataset, verbose=False) -> Dict[int, Dict]:
        """Distribute MNIST data in IID manner among clients"""
        # Load MNIST dataset
        #train_dataset, test_dataset = self.load_mnist_data(data_dir)
        
        # Calculate samples per client
        train_samples_per_client = len(train_dataset) // self.num_clients
        test_samples_per_client = len(test_dataset) // self.num_clients
        
        # Randomly shuffle indices
        train_indices = torch.randperm(len(train_dataset)).tolist()
        test_indices = torch.randperm(len(test_dataset)).tolist()
        
        # Distribute data to clients
        for client_id in range(self.num_clients):
            # Calculate start and end indices for this client
            train_start = client_id * train_samples_per_client
            train_end = (client_id + 1) * train_samples_per_client
            
            test_start = client_id * test_samples_per_client
            test_end = (client_id + 1) * test_samples_per_client
            
            # Handle remainder for last client
            if client_id == self.num_clients - 1:
                train_end = len(train_dataset)
                test_end = len(test_dataset)
            
            # Get client's data indices
            client_train_indices = train_indices[train_start:train_end]
            client_test_indices = test_indices[test_start:test_end]
            
            # Create subset datasets for this client
            client_train_dataset = torch.utils.data.Subset(train_dataset, client_train_indices)
            client_test_dataset = torch.utils.data.Subset(test_dataset, client_test_indices)

            # Print the label distribution for this client
            train_labels = [train_dataset.targets[i] for i in client_train_indices]
            test_labels = [test_dataset.targets[i] for i in client_test_indices]
            
            # Labels to int
            train_labels = [int(label) for label in train_labels]
            test_labels = [int(label) for label in test_labels]
            
            if self.verbose:
                print(f"Client {client_id} - Train labels: {Counter(train_labels)}")
                print(f"Client {client_id} - Test labels: {Counter(test_labels)}")
                # Total samples
                print(f"Client {client_id} - Total train samples: {len(client_train_dataset)}")
                print(f"Client {client_id} - Total test samples: {len(client_test_dataset)}")
                print()

            # Create data loaders
            train_loader = DataLoader(
                client_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False
            )
            
            test_loader = DataLoader(
                client_test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )
            
            # Store client data information
            self.client_data[client_id] = {
                'train_dataset': client_train_dataset,
                'test_dataset': client_test_dataset,
                'train_samples': len(client_train_dataset),
                'test_samples': len(client_test_dataset),
                'train_indices': client_train_indices,
                'test_indices': client_test_indices
            }
            
            self.client_loaders[client_id] = {
                'train_loader': train_loader,
                'test_loader': test_loader
            }
        

    def _distribute_dirichlet_data(self, train_dataset: Dataset, test_dataset: Dataset, verbose=False) -> Dict[int, Dict]:
        """Distribute MNIST data in a Non-IID manner using Dirichlet distribution for training,
        and uniformly (IID) for test set."""
        # Train labels
        train_labels = np.array(train_dataset.targets)

        # Group train indices by class
        train_indices_by_class = {i: np.where(train_labels == i)[0] for i in range(10)}

        # Init client storage
        for client_id in range(self.num_clients):
            self.client_data[client_id] = {
                'train_indices': [],
                'test_indices': []
            }

        # --- Dirichlet partition for training set ---
        for c in range(10):  # for each class
            idxs = train_indices_by_class[c]
            np.random.shuffle(idxs)

            # Dirichlet proportions
            proportions = np.random.dirichlet(alpha=np.repeat(self.alpha, self.num_clients))

            # Split according to proportions
            split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            split_indices = np.split(idxs, split_points)

            for client_id, client_split in enumerate(split_indices):
                self.client_data[client_id]['train_indices'].extend(client_split.tolist())

        # --- Uniform partition for test set ---
        test_indices = torch.randperm(len(test_dataset)).tolist()
        test_samples_per_client = len(test_dataset) // self.num_clients

        for client_id in range(self.num_clients):
            start = client_id * test_samples_per_client
            end = (client_id + 1) * test_samples_per_client if client_id != self.num_clients - 1 else len(test_dataset)

            self.client_data[client_id]['test_indices'] = test_indices[start:end]

        # --- Build loaders and datasets ---
        test_labels = np.array(test_dataset.targets)
        for client_id in range(self.num_clients):
            client_train_dataset = torch.utils.data.Subset(train_dataset, self.client_data[client_id]['train_indices'])
            client_test_dataset = torch.utils.data.Subset(test_dataset, self.client_data[client_id]['test_indices'])

            # Print label distribution
            train_lbls = [int(train_labels[i]) for i in self.client_data[client_id]['train_indices']]
            test_lbls = [int(test_labels[i]) for i in self.client_data[client_id]['test_indices']]
            
            if self.verbose:
                print(f"Client {client_id} - Train labels: {Counter(train_lbls)}")
                print(f"Client {client_id} - Test labels: {Counter(test_lbls)}")
                # Total samples
                print(f"Client {client_id} - Total train samples: {len(client_train_dataset)}")
                print(f"Client {client_id} - Total test samples: {len(client_test_dataset)}")
                print()

            # Create data loaders
            train_loader = DataLoader(client_train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
            test_loader = DataLoader(client_test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

            # Store info
            self.client_data[client_id].update({
                'train_dataset': client_train_dataset,
                'test_dataset': client_test_dataset,
                'train_samples': len(client_train_dataset),
                'test_samples': len(client_test_dataset)
            })
            self.client_loaders[client_id] = {
                'train_loader': train_loader,
                'test_loader': test_loader
            }

        #return self.client_data

    def get_data_summary(self) -> Dict:
        """Get summary of data distribution"""
        if not self.client_data:
            return {}
        
        total_train = sum(client['train_samples'] for client in self.client_data.values())
        total_test = sum(client['test_samples'] for client in self.client_data.values())
        
        return {
            'name': self.name,
            'partition': f"dirichlet(alpha={self.alpha})" if self.partition == 'dirichlet' else self.partition,
            'total_clients': self.num_clients,
            'total_train_samples': total_train,
            'total_test_samples': total_test,
            'avg_train_per_client': total_train // self.num_clients,
            'avg_test_per_client': total_test // self.num_clients
        }
    
    def get_client_loader(self, client_id: int, loader_type: str = 'train') -> DataLoader:
        """Get data loader for a specific client"""
        if client_id not in self.client_loaders:
            raise ValueError(f"Client {client_id} not found. Available clients: {list(self.client_loaders.keys())}")
        
        if loader_type not in ['train', 'test']:
            raise ValueError("loader_type must be 'train' or 'test'")
        
        return self.client_loaders[client_id][f'{loader_type}_loader']

    
    
# --- Custom Dataset Class for Shakespeare ---
class ShakespeareDataset(Dataset):
    """Custom PyTorch Dataset for Shakespeare Text"""
    def __init__(self, data: List[np.ndarray], seq_len: int = 80):
        self.data = data  # List of integer arrays (one per client/role)
        self.seq_len = seq_len
        self.samples = []
        
        # Pre-calculate valid start indices to create samples
        # Each client's data is treated independently
        for client_idx, client_data in enumerate(self.data):
            # We need at least seq_len + 1 (input + target)
            if len(client_data) <= self.seq_len:
                continue
                
            # Create indices for this client
            # Sample format: (client_index, start_index)
            # Stride of 1 for maximum samples, or higher to reduce overlap
            for i in range(0, len(client_data) - self.seq_len, 1): 
                self.samples.append((client_idx, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        client_idx, start_idx = self.samples[idx]
        client_data = self.data[client_idx]
        
        # Input: seq_len characters
        # Target: same sequence shifted by 1 character
        chunk = client_data[start_idx : start_idx + self.seq_len + 1]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y