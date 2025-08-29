import numpy as np
import networkx as nx
from scipy.linalg import eigh

eigvals, eigvecs = eigh(L)



def selection_embedding(k=5, B=3, sigma=1.0):
    # build embedding (skip trivial eigenvector at index 0)
    U = eigvecs[:, 1:k+1]
    neighbors = {}
    for i in range(n):
        dists = np.linalg.norm(U - U[i], axis=1)**2
        probs = np.exp(-dists / sigma**2)
        probs[i] = 0  # exclude self
        probs /= probs.sum()
        chosen = np.random.choice(n, size=B, replace=False, p=probs)
        neighbors[i] = chosen
    return neighbors

def selection_heat_kernel(t=1.0, k=20, B=3):
    # truncate spectrum
    vals = eigvals[:k]
    vecs = eigvecs[:, :k]
    K = vecs @ np.diag(np.exp(-t*vals)) @ vecs.T  # heat kernel
    neighbors = {}
    for i in range(n):
        probs = K[i].copy()
        probs[i] = 0
        probs = probs / probs.sum()
        chosen = np.random.choice(n, size=B, replace=False, p=probs)
        neighbors[i] = chosen
    return neighbors
