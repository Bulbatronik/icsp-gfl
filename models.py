import torch
import torch.nn as nn
import torch.functional as F


def load_model(model_name, device=None):
    if model_name == "mnistCNN":
        return MNIST_cnn(device=device)
    elif model_name == "mnistMLP":
        return MNIST_mlp(device=device)
    elif model_name == "cifar10CNN":
        return CIFAR10_cnn(device=device)
    else:
        raise ValueError(f"Model {model_name} not recognized.")


class MNIST_mlp(nn.Module):
    """Lightweight CNN for MNIST"""
    def __init__(self, device=None):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        
        # Set device with better error handling
        try:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            elif isinstance(device, str):
                self.device = torch.device(device)
                if self.device.type == 'cuda' and not torch.cuda.is_available():
                    print("\033[93mWarning: CUDA requested for model but not available. Falling back to CPU.\033[0m")
                    self.device = torch.device('cpu')
            else:
                self.device = device
            self.to(self.device)
        except Exception as e:
            print(f"\033[91mError setting device for model: {e}. Falling back to CPU.\033[0m")
            self.device = torch.device('cpu')
            self.to(self.device)
        
    def forward(self, x):
        # Move input to the device if it's not already there
        if x.device != self.device:
            x = x.to(self.device)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MNIST_cnn(nn.Module):
    """Lightweight CNN for MNIST"""
    def __init__(self, device=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # Set device with better error handling
        try:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            elif isinstance(device, str):
                self.device = torch.device(device)
                if self.device.type == 'cuda' and not torch.cuda.is_available():
                    print("\033[93mWarning: CUDA requested for model but not available. Falling back to CPU.\033[0m")
                    self.device = torch.device('cpu')
            else:
                self.device = device
            self.to(self.device)
        except Exception as e:
            print(f"\033[91mError setting device for model: {e}. Falling back to CPU.\033[0m")
            self.device = torch.device('cpu')
            self.to(self.device)
        
    def forward(self, x):
        # Move input to the device if it's not already there
        if x.device != self.device:
            x = x.to(self.device)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    
class CIFAR10_cnn(nn.Module):
    """Lightweight CNN for MNIST"""
    def __init__(self, device=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.fc = nn.Linear(64 * 4 * 4, 10)
        # Set device with better error handling
        try:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            elif isinstance(device, str):
                self.device = torch.device(device)
                if self.device.type == 'cuda' and not torch.cuda.is_available():
                    print("\033[93mWarning: CUDA requested for model but not available. Falling back to CPU.\033[0m")
                    self.device = torch.device('cpu')
            else:
                self.device = device
            self.to(self.device)
        except Exception as e:
            print(f"\033[91mError setting device for model: {e}. Falling back to CPU.\033[0m")
            self.device = torch.device('cpu')
            self.to(self.device)
        
    def forward(self, x):
        # Move input to the device if it's not already there
        if x.device != self.device:
            x = x.to(self.device)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
