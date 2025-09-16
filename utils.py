import torch
import numpy as np
import random


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
def experiment_name(cfg):
    """Generate a descriptive experiment name based on the configuration"""
    topology_part = f"topo-{cfg['network']['topology']}"
    topology_part += "_"
    topology_part += f"C-{cfg['network']['num_clients']}"
    if cfg['network']['topology'] == "random":
        topology_part += "_"
        topology_part += f"p-{cfg['network']['edge_proba']}"
    elif cfg['network']['topology'] == "smallwrld":
        topology_part += "_"
        topology_part += f"k{cfg['network']['k']}"
        topology_part += "_"
        topology_part += f"p{cfg['network']['p']}"

    if cfg['client']['selection_ratio'] == 1.0:
        selection_part = "broadcast"
    else:
        selection_part = f"select-{cfg['client']['selection_method']}"
        selection_part += "_"
        if cfg['client']['selection_method'] == "spectrclust":
            selection_part += f"Neig-{cfg['client']['num_eig']}"
            selection_part += "_"
            selection_part += f"dist-{cfg['client']['dist']}"
            selection_part += "_"
        elif cfg['client']['selection_method'] == "heatkernel":
            selection_part += f"t-{cfg['client']['t']}"
            selection_part += "_"
        selection_part += f"ratio-{cfg['client']['selection_ratio']}"
        selection_part += "_"
        selection_part += f"tau-{cfg['client']['tau']}"
        
    dataset_part = f"D-{cfg['dataset']['name']}"
    dataset_part += "_"
    dataset_part += f"split-{cfg['dataset']['partition']}"
    if cfg['dataset']['partition'] == 'dirichlet':
        dataset_part += "_"
        dataset_part += f"a-{cfg['dataset']['alpha']}"
        
    federation_part = f"R-{cfg['federation']['rounds']}"
    federation_part += "_"
    federation_part += f"patA-{cfg['federation']['rounds_patience_acc']}"
    federation_part += "_"
    federation_part += f"patL-{cfg['federation']['rounds_patience_loss']}"
    
    client_part = f"M-{cfg['training']['architecture']}"
    client_part += "_"
    client_part += f"O-{cfg['training']['optimizer']}"
    client_part += "_"
    client_part += f"lr-{cfg['training']['lr']}"
    client_part += "_"
    client_part += f"E-{cfg['training']['epochs']}"
    client_part += "_"
    client_part += f"B-{cfg['dataset']['batch_size']}"
    client_part += "_"
    client_part += f"rho-{cfg['training']['rho']}"
    
    name_parts = [
        topology_part,
        selection_part,
        dataset_part,
        federation_part,
        client_part,
        f"s-{cfg['seed']}"
        ]
    return "_".join(name_parts)