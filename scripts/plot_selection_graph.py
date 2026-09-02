#!/usr/bin/env python
"""Regenerate the "selection-probability" graph figures (e.g. Fig. 1 in the
paper: ``small_spectr.png`` / ``small_heatkern.png``) *without* running the
training pipeline.

It reuses the exact code path from ``main.py``:

    topology -> DecentralizedClient(...) -> constr_prob_matrix -> plot_transition_graph

so the per-node transmission probabilities are computed identically to a real
run. Only the topology-only selectors (spectral embedding, heat kernel,
broadcast, random, connectivity-aware) are meaningful here, because they depend
solely on the graph Laplacian. The ``gradients`` and ``kld`` (Data Sim.)
selectors are data-/gradient-dependent and only take shape during training, so
they are rejected with a clear message.

Node positions are computed once (via the same ``plot_topology`` layout the
pipeline uses) and shared across all requested methods, so the panels are
directly comparable.

Examples
--------
Reproduce the two small-graph panels used in the paper::

    python scripts/plot_selection_graph.py --topology small --methods spectral heat

Custom heat-kernel diffusion time and embedding dimension::

    python scripts/plot_selection_graph.py --topology women --methods spectral heat \
        --num-eig 6 --t 0.05
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch import nn

# Make sure the repo root (which holds topology.py, client.py, ...) is importable
# regardless of the current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from topology import NetworkTopology
from client import DecentralizedClient, constr_prob_matrix
from visualize import plot_topology, plot_transition_graph
from utils import set_seed

# Friendly aliases -> the selection_method strings understood by DecentralizedClient
METHOD_ALIASES = {
    "spectral": "spectrclust", "spectr": "spectrclust", "spectrclust": "spectrclust",
    "heat": "heatkernel", "heatkern": "heatkernel", "heatkernel": "heatkernel",
    "broadcast": "broadcast",
    "random": "random",
    "connaware": "connaware", "resistance": "connaware",
}
# Short filename suffixes matching the paper's figure names (small_spectr / small_heatkern)
METHOD_SUFFIX = {
    "spectrclust": "spectr",
    "heatkernel": "heatkern",
    "broadcast": "broadcast",
    "random": "random",
    "connaware": "connaware",
}
# These need per-round data/gradients and cannot be drawn as a static topology figure.
DATA_DEPENDENT = {"gradients", "kld"}


def build_clients(G, method, args, num_clients):
    """Instantiate one DecentralizedClient per node, exactly as main.py does.

    A throw-away 1-parameter model and an empty label distribution are supplied:
    the transmission probabilities for topology-only selectors depend only on the
    graph, so these placeholders never influence the result.
    """
    dummy_model = nn.Linear(1, 1)                       # cheapest valid model for the optimizer
    dummy_loader = torch.utils.data.DataLoader(         # only len(.dataset) is read
        torch.utils.data.TensorDataset(torch.zeros(1, 1), torch.zeros(1)), batch_size=1
    )
    label_dist = np.zeros((num_clients, 10), dtype=float)  # unused by topology-only selectors

    clients = {}
    for cid in range(num_clients):
        clients[cid] = DecentralizedClient(
            client_id=cid,
            graph=G,
            model=nn.Linear(1, 1) if cid else dummy_model,
            train_loader=dummy_loader,
            test_loader=dummy_loader,
            selection_method=method,
            num_eig=args.num_eig,
            t=args.t,
            tau=args.tau,
            selection_ratio=args.ratio,
            dist=args.dist,
            optimizer="SGD",
            epochs=1,
            lr=0.01,
            rho=0.9,
            device="cpu",
            label_dist=label_dist,
        )
    return clients


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topology", default="small",
                   help="Graph: small, women, miserable, karate, random, smallwrld, ... (default: small)")
    p.add_argument("--num-clients", type=int, default=8,
                   help="Node count for size-parametric topologies (small=8 fixed; "
                        "women/miserable/karate override this). Default: 8")
    p.add_argument("--methods", nargs="+", default=["spectral", "heat"],
                   help="One or more of: spectral, heat, broadcast, random, connaware "
                        "(default: spectral heat)")
    p.add_argument("--num-eig", type=int, default=3,
                   help="Embedding dimension k / number of Laplacian eigenvectors (default: 3; "
                        "paper uses 3/6/12 for small/medium/large)")
    p.add_argument("--t", type=float, default=0.01,
                   help="Heat-kernel diffusion time (default: 0.01, the paper's small-graph value)")
    p.add_argument("--tau", type=float, default=1.0, help="Softmax temperature (default: 1.0)")
    p.add_argument("--ratio", type=float, default=0.5, help="Subsampling ratio beta (default: 0.5)")
    p.add_argument("--dist", default="cosine", choices=["cosine", "eucl"],
                   help="Spectral-embedding distance (default: cosine)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the (layout) RNG so panels are reproducible (default: 42)")
    p.add_argument("--layout", default="auto",
                   help="networkx layout: auto, spring, circular, shell, spectral, kamada_kawai")
    p.add_argument("--edge-width", type=float, default=5.0,
                   help="Thickness of the graph links (default: 5.0; the adaptive default is ~2.5)")
    p.add_argument("--edge-proba", type=float, default=0.4, help="Edge prob. for 'random' topology")
    p.add_argument("--k", type=int, default=4, help="k for 'smallwrld' topology")
    p.add_argument("--p", type=float, default=0.3, help="Rewire prob. for 'smallwrld' topology")
    p.add_argument("--outdir", default=os.path.join(REPO_ROOT, "figures_selection"),
                   help="Output folder; figures are written to <outdir>/plots/ (default: ./figures_selection)")
    p.add_argument("--name-prefix", default=None,
                   help="Filename prefix (default: the topology name, giving e.g. small_spectr.png)")
    args = p.parse_args()

    # Resolve & validate methods up front.
    methods = []
    for m in args.methods:
        key = m.lower()
        if key in DATA_DEPENDENT:
            p.error(f"'{m}' is data-/gradient-dependent and only forms during training; "
                    f"it cannot be rendered as a static topology figure.")
        if key not in METHOD_ALIASES:
            p.error(f"Unknown method '{m}'. Choose from: {', '.join(sorted(set(METHOD_ALIASES)))}.")
        methods.append(METHOD_ALIASES[key])

    # Seed BEFORE building the graph/layout so the shared node positions are reproducible,
    # mirroring set_seed(cfg['seed']) at the top of main.py.
    set_seed(args.seed)

    # 1) Build the topology (same class the pipeline uses).
    network = NetworkTopology(num_clients=args.num_clients, topology=args.topology,
                              edge_proba=args.edge_proba, k=args.k, p=args.p)
    network.create_topology()
    G = network.G
    num_clients = network.num_clients  # women/miserable/karate reset this to the real node count
    prefix = args.name_prefix or args.topology

    # 2) Compute node positions ONCE and reuse them for every method (as main.py does:
    #    pos = plot_topology(...) is computed once, then passed to plot_transition_graph).
    pos = plot_topology(G, layout_type=args.layout, save_folder=args.outdir,
                        file_name=f"{prefix}_topology")

    # 3) For each method: build clients -> probability matrix -> selection graph figure.
    saved = []
    for method in methods:
        clients = build_clients(G, method, args, num_clients)
        P = constr_prob_matrix(clients)
        fname = f"{prefix}_{METHOD_SUFFIX[method]}"
        plot_transition_graph(P, pos, save_folder=args.outdir, file_name=fname,
                              edge_width=args.edge_width)
        saved.append(os.path.join(args.outdir, "plots", f"{fname}.png"))

    print("\nSaved figures:")
    for pth in saved:
        print(f"  {pth}")
    print("\nTo drop them into the paper, copy the PNGs into paper/figures/, e.g.:")
    for pth in saved:
        print(f"  cp {pth} paper/figures/{os.path.basename(pth)}")


if __name__ == "__main__":
    main()
