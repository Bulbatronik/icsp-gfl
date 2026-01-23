import torch
import torch.nn as nn
import torch.functional as F


def load_model(model_name, vocab_size=None, device=None):
    if model_name == "mnistCNN":
        return MNIST_cnn(device=device)
    elif model_name == "mnistMLP":
        return MNIST_mlp(device=device)
    elif model_name == "cifar10CNN":
        return CIFAR10_cnn(device=device)
    elif model_name == "shakespeareLSTM":
        return ShakespeareLSTM(vocab_size=vocab_size, device=device)
    else:
        raise ValueError(f"Model {model_name} not recognized.")


class MNIST_mlp(nn.Module):
    """Lightweight CNN for MNIST"""
    def __init__(self, device=None):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
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
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

class MNIST_cnn(nn.Module):
    """Lightweight CNN for MNIST"""
    def __init__(self, device=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.relu3 = nn.ReLU()
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
            
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)
    
    
class CIFAR10_cnn(nn.Module):
    """Lightweight CNN for MNIST"""
    def __init__(self, device=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.relu3 = nn.ReLU()
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
            
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size, device=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 8)
        self.lstm = nn.LSTM(8, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, vocab_size)
        self.vocab_size = None
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
        
    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = out.reshape(-1, out.shape[2]) # Flatten for FC
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(NUM_LAYERS, batch_size, HIDDEN_DIM).zero_(),
                weight.new(NUM_LAYERS, batch_size, HIDDEN_DIM).zero_())