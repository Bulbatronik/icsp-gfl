#!/usr/bin/env python3
"""Parse results/<exp>/results.json into a compact table.

Final accuracy = mean test_accuracy over the last `--last` rounds.
Usage: python scripts/parse_results.py [--filter SUBSTR] [--last 5]
"""
import argparse, json, glob, os, re

def short_method(name):
    if "broadcast" in name: return "broadcast"
    for m in ("nofed", "random", "gradients", "spectrclust", "heatkernel"):
        if f"select-{m}" in name: return m
    return "?"

def parse(args):
    rows = []
    for path in sorted(glob.glob("results/*/results.json")):
        exp = os.path.basename(os.path.dirname(path))
        if args.filter and args.filter not in exp:
            continue
        with open(path) as f:
            data = json.load(f)
        if not data:
            continue
        accs = [r["test_accuracy"] for r in data[-args.last:]]
        final = sum(accs) / len(accs)
        net = re.search(r"topo-([a-z]+)_C-(\d+)", exp)
        split = re.search(r"split-(\w+?)_R-", exp)
        keig = re.search(r"Neig-(\d+)", exp)
        t = re.search(r"_t-([\d.]+)_", exp)
        ratio = re.search(r"ratio-([\d.]+)", exp)
        rows.append({
            "net": net.group(1) if net else "?",
            "C": net.group(2) if net else "?",
            "split": split.group(1) if split else "?",
            "method": short_method(exp),
            "k": keig.group(1) if keig else "-",
            "t": t.group(1) if t else "-",
            "beta": ratio.group(1) if ratio else ("1.0" if "broadcast" in exp else "-"),
            "acc": round(final, 2),
            "rounds": len(data),
        })
    hdr = ["net", "C", "split", "method", "k", "t", "beta", "rounds", "acc"]
    print("  ".join(f"{h:>10}" for h in hdr))
    for r in sorted(rows, key=lambda x: (x["net"], x["split"], x["method"])):
        print("  ".join(f"{str(r[h]):>10}" for h in hdr))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="")
    ap.add_argument("--last", type=int, default=5)
    parse(ap.parse_args())
