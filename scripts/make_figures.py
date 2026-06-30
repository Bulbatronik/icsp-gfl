#!/usr/bin/env python3
"""Build the results figure for the paper from results/<exp>/results.json.

Panels:
  (a) convergence on the community-correlated (block) regime, women graph;
  (b) accuracy vs sampling ratio beta (mean +/- std over seeds);
  (c) accuracy vs embedding dimension k (mean +/- std over seeds).
Saved to paper/figures/sensitivity.png.
"""
import json, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")

def curves(must, nots=()):
    out = []
    for p in glob.glob(os.path.join(RES, "*/results.json")):
        e = os.path.basename(os.path.dirname(p))
        if all(s in e for s in must) and not any(s in e for s in nots):
            out.append([r["test_accuracy"] for r in json.load(open(p))])
    return out

def finals(must, nots=(), last=10):
    return [sum(c[-last:]) / len(c[-last:]) for c in curves(must, nots)]

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "lines.linewidth": 1.3, "lines.markersize": 4, "legend.fontsize": 6})
fig, ax = plt.subplots(1, 3, figsize=(7.4, 1.95))

# ---- (a) convergence, miserable (N=77), strong heterogeneity (Dirichlet alpha=0.1) ----
conv = {"Broadcast": (["topo-miserable", "broadcast", "split-dir", "R-100"], "k", "-"),
        "Spectr. Emb.": (["topo-miserable", "select-spectrclust", "split-dir", "R-100"], "C0", "-"),
        "Heat Kern.": (["topo-miserable", "select-heatkernel", "t-0.05_", "split-dir", "R-100"], "C2", "-."),
        "Grad. Sim.": (["topo-miserable", "select-gradients", "split-dir", "R-100"], "C4", (0, (3, 1, 1, 1))),
        "Random": (["topo-miserable", "select-random", "split-dir", "R-100"], "C1", "--"),
        "Data Sim.": (["topo-miserable", "select-kld", "split-dir", "R-100"], "C3", ":"),
        "No-Fed.": (["topo-miserable", "select-nofed", "split-dir", "R-100"], "0.5", ":")}
for name, (must, col, ls) in conv.items():
    cs = curves(must)
    if cs:
        L = min(map(len, cs)); arr = np.array([c[:L] for c in cs])
        m = arr.mean(axis=0); x = range(1, L + 1)
        ax[0].plot(x, m, color=col, linestyle=ls, label=name)
        if len(cs) > 1:  # std band for the multi-seed methods (spectral, random)
            s = arr.std(axis=0)
            ax[0].fill_between(x, m - s, m + s, color=col, alpha=0.18, linewidth=0)
ax[0].set_xlabel("Communication round"); ax[0].set_ylabel("Test accuracy (%)")
ax[0].set_title(r"(a) large graph, $\alpha{=}0.1$"); ax[0].legend(loc="lower right")

# ---- (b) accuracy vs beta (small, Dirichlet alpha=0.3), mean +/- std ----
betas = [0.25, 0.5, 0.75]
def stat(fn): a = np.array(fn); return (np.mean(a), np.std(a)) if len(a) else (np.nan, 0)
sp = [stat(finals(["topo-small", "select-spectrclust", "split-dir", "Neig-3_", f"ratio-{b}"], nots=("smallwrld",))) for b in betas]
rd = [stat(finals(["topo-small", "select-random", "split-dir", f"ratio-{b}"], nots=("smallwrld",))) for b in betas]
#ax[1].errorbar(betas, [m for m, _ in sp], yerr=[s for _, s in sp], fmt="o-", color="C0", capsize=2, label="Spectr. Emb.")
#ax[1].errorbar(betas, [m for m, _ in rd], yerr=[s for _, s in rd], fmt="s--", color="C1", capsize=2, label="Random")
ax[1].errorbar(betas, [m for m, _ in rd], yerr=[s for _, s in rd], fmt="o-", color="C0", capsize=2, label="Spectr. Emb.")
ax[1].errorbar(betas, [m for m, _ in sp], yerr=[s for _, s in sp], fmt="s--", color="C1", capsize=2, label="Random")

ax[1].set_xlabel(r"Sampling ratio $\beta$")
ax[1].set_title("(b) ratio sensitivity"); ax[1].legend(loc="upper left")

# ---- (c) accuracy vs k (small, Dirichlet alpha=0.3, beta=0.5), mean +/- std ----
ks = [2, 3, 5, 6, 7]
ak = [stat(finals(["topo-small", "select-spectrclust", "split-dir", f"Neig-{k}_", "ratio-0.5"], nots=("smallwrld",))) for k in ks]
ax[2].errorbar(ks, [m for m, _ in ak], yerr=[s for _, s in ak], fmt="^-", color="C0", capsize=2)
ax[2].set_xlabel(r"Embedding dim. $k$")
ax[2].set_title("(c) dimension sensitivity"); ax[2].set_xticks(ks)

fig.tight_layout(pad=0.5, w_pad=1.8)
out = os.path.join(ROOT, "paper", "figures", "sensitivity.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
for name, (must, _, _) in conv.items():
    cs = curves(must); print(name, "n_seeds=", len(cs), "final=", [round(c[-1], 1) for c in cs])
print("beta spectr:", [(round(m,2),round(s,2)) for m,s in sp]); print("beta random:", [(round(m,2),round(s,2)) for m,s in rd])
print("k:", [(round(m,2),round(s,2)) for m,s in ak])
