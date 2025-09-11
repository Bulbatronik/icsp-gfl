import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Dict


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
        
    
    def load_and_distribute(self):
        # Load the dataset 
        if self.name == 'mnist':
            train_dataset, test_dataset = self._load_mnist_data()
        elif self.name == 'cifar10':
            train_dataset, test_dataset = self._load_cifar10_data()
        elif self.name == 'fmnist':
            train_dataset, test_dataset = self._load_fashion_mnist_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.name}. Supported: 'mnist', 'cifar10'")
        
        # Partition the data among the clients
        if self.partition == 'iid':
            self._distribute_iid_data(train_dataset, test_dataset)
        elif self.partition == 'dir':
            self._distribute_dirichlet_data(train_dataset, test_dataset)
        else:
            raise ValueError(f"Unsupported partitioning method: {self.partition}. Supported: 'iid', 'dir'")
        
        
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

    