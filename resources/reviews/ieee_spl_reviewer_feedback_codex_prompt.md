# Codex Prompt: Revise IEEE SPL Paper Based on Reviewer Feedback

## Role

You are working inside the repository for the paper **“Connectivity-Aware Client Selection in Decentralized Federated Learning.”** Your task is to revise the manuscript and, where feasible, update the experimental code to address the IEEE Signal Processing Letters reviewer feedback.

The goal is not to superficially respond to reviewers. The goal is to make the paper technically stronger, more precise, and better supported by experiments.

## Absolute formatting requirement: every manuscript change must be blue

All changes to the manuscript **must appear in blue in the compiled PDF**.

Assume the manuscript is written in LaTeX.

1. Inspect the main `.tex` file and preamble.
2. If `xcolor` is not already available, add:

   ```latex
   \usepackage{xcolor}
   ```

3. Add a revision macro if none exists:

   ```latex
   \newcommand{\revise}[1]{\textcolor{blue}{#1}}
   ```

4. Wrap every added or modified manuscript sentence, phrase, theorem/claim wording, equation explanation, figure caption, table caption, table entry, and result interpretation in `\revise{...}`.
5. Do **not** color unchanged text.
6. Do **not** make silent uncolored manuscript edits.
7. For changed equations, either color the changed equation terms directly or color the explanatory text immediately surrounding them.
8. For new or changed tables, color added/modified entries with `\textcolor{blue}{...}` where LaTeX permits.
9. For new bibliography entries:
   - Add BibTeX entries normally if the project uses `.bib`.
   - Mark the manuscript text introducing and citing those works in blue.
   - Do not break IEEE bibliography style merely to color generated references.
10. Add a short internal revision log, either as `revision_log.md` or as comments in the manuscript source, listing every substantive manuscript and experiment change.

Do not claim an experiment was run unless the repository contains the resulting logs, figures, tables, or saved outputs.

## Reviewer-driven revision objectives

The reviewers found the paper clear, simple, and promising, but they questioned novelty, graph assumptions, topology-only relevance, spectral design choices, experimental coverage, and scalability. Revise the paper so that these concerns are directly addressed.

The main revision should do the following:

1. **Tone down contribution claims.**
   - Do not present the method as a fundamentally new spectral graph method.
   - Reframe it as a communication-efficient topology-aware client selection framework for DFL.
   - Explicitly state that the method adapts established Laplacian embedding and heat-kernel diffusion tools to client selection under a fixed communication topology.
   - Avoid overstating novelty, decentralization, scalability, or learning optimality.

2. **Clarify graph construction.**
   - Add a dedicated paragraph or subsection explaining what the communication graph represents.
   - Distinguish between:
     - connectivity/availability-based graphs,
     - latency- or physical-proximity-based graphs,
     - infrastructure-imposed graphs,
     - data/model-similarity-derived graphs.
   - State clearly that if the graph is derived from data or model similarity, the method is no longer purely topology-only.
   - State the paper’s default assumption: the graph is fixed before training and reflects communication feasibility, not private data or gradients.
   - Discuss when topology may or may not correlate with learning relevance.

3. **Address topology-only limitations.**
   - Add a limitations paragraph explaining that structural similarity alone may be insufficient under strong statistical heterogeneity.
   - Explain that neighboring or structurally close clients can have very different label distributions.
   - Explicitly note that topology-only selection trades off learning-awareness for lower communication and privacy overhead.
   - Add a possible hybrid extension that combines topology-aware scores with lightweight learning-aware signals, without claiming it is implemented unless it is.

4. **Clarify decentralization and spectral computation.**
   - The reviewers object that eigenvectors and heat kernels require global topology knowledge.
   - Add a paragraph explaining how spectral quantities are obtained.
   - Be explicit:
     - If computed centrally before training, say so.
     - If computed once from a public or known communication graph, say so.
     - If using distributed eigensolvers or local approximations, describe them only if implemented or clearly presented as future work.
   - Remove or qualify any claim that the method is fully decentralized if global topology preprocessing is required.
   - Add computational complexity discussion for Laplacian eigendecomposition, heat-kernel computation, and per-round neighbor sampling.

5. **Strengthen discussion of spectral design.**
   - Justify why low-frequency Laplacian eigenvectors are used.
   - Acknowledge that low-frequency eigenvectors emphasize smooth structural variation and may be insufficient for heterophilous or label-mismatched neighborhoods.
   - Add discussion of alternatives:
     - mixed low/high-frequency spectral features,
     - full-spectrum Laplacian positional encodings,
     - learned spectral encodings,
     - heat-kernel scales.
   - If feasible, add an ablation comparing low-frequency embeddings against mixed-frequency or higher-frequency alternatives.

6. **Expand experiments if code and resources permit.**
   - Add experiments that directly answer reviewer concerns.
   - Do not fabricate results.
   - If full experiments cannot be run, implement configs/scripts and document what remains to be run.

7. **Add stronger baselines.**
   - At minimum, compare against:
     - uniform random neighbor sampling,
     - full-neighbor DFL aggregation,
     - degree- or centrality-based neighbor selection,
     - one learning-aware or gradient/model-based selection baseline, if implementable.
   - Consider adapting:
     - gradient/model-update similarity selection,
     - loss-based or validation-performance-based selection,
     - node-selection methods from FL to the DFL setting,
     - Laplacian matrix sampling for decentralized learning if implementation is feasible.
   - If a suggested baseline is not included, add a concise explanation of why it is not directly comparable or not feasible within the letter’s scope.

8. **Add ablations and sensitivity analysis.**
   - Run or implement ablations for:
     - embedding dimension `k`,
     - diffusion time `t`,
     - sampling ratio `beta`,
     - graph size `N`,
     - heterogeneity level, including several Dirichlet `alpha` values.
   - Use at least three random seeds where feasible.
   - Report mean and standard deviation.

9. **Evaluate larger and more realistic topologies.**
   - Add experiments or scripts for larger graphs, for example `N = 100`, `N = 500`, and `N = 1000` where feasible.
   - Include realistic graph families when possible:
     - random geometric graphs,
     - Watts–Strogatz small-world graphs,
     - Barabási–Albert scale-free graphs,
     - stochastic block/community graphs,
     - sparse infrastructure-like graphs.
   - Report whether performance gains persist as graph size increases.

10. **Test topology/data mismatch and heterophily.**
    - Add an experiment where data heterogeneity is independent of topology.
    - Add an experiment where topology is correlated with client data distribution.
    - Add a hard case where structurally close clients have dissimilar distributions.
    - Compare low-frequency spectral selection under these settings.
    - Discuss failure modes honestly.

11. **Analyze dynamic or time-varying topology.**
    - Add either experiments or a limitation/future-work section on:
      - node dropout,
      - edge dropout,
      - random rewiring,
      - client churn,
      - stale spectral embeddings.
    - If feasible, run a simple dynamic-topology experiment with periodic edge rewiring or random node dropout.

12. **Add communication–accuracy and overhead analysis.**
    - Report:
      - test accuracy versus communication rounds,
      - total transmitted bytes to reach a target accuracy,
      - communication per round,
      - selection/preprocessing time,
      - eigendecomposition or heat-kernel computation time,
      - memory overhead for storing spectral embeddings or diffusion scores.
    - Add a table summarizing computational complexity.

13. **Update related work.**
    - Add and discuss the references suggested by Reviewer 2.
    - Use full bibliographic information and DOI where available.
    - Suggested references from the review:
      - Alshami et al., “SCDFL: A spectral clustering-based framework for accelerating convergence in decentralized federated learning,” *Computer Networks*, 2025, article 111615.
      - Chiu et al., “Laplacian matrix sampling for communication-efficient decentralized learning,” *IEEE Journal on Selected Areas in Communications*, vol. 41, no. 4, pp. 887–901, 2023.
      - Liu et al., “Accelerating decentralized federated learning with probabilistic communication in heterogeneous edge computing,” *IEEE Transactions on Networking*, 2025.
      - Ito, Zhu, Chen, Koutra, and Wiens, “Learning Laplacian positional encodings for heterophilous graphs,” AISTATS 2025.
      - Guo et al., “Byzantine-resilient decentralized stochastic gradient descent,” *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 32, no. 6, pp. 4096–4106, 2021.
      - Wu and Wang, “Node selection toward faster convergence for federated learning on non-IID data,” *IEEE Transactions on Network Science and Engineering*, vol. 9, no. 5, pp. 3099–3111, 2022.
    - Do not merely list these papers. Explain how they differ from the proposed method and how they affect the novelty claim.

## Concrete manuscript edits to make

### Abstract

Revise the abstract to:
- reduce novelty claims,
- state the fixed-graph assumption,
- state that the method uses topology-only spectral scores,
- mention the trade-off between communication reduction and learning-aware selection,
- avoid implying that topology alone is always sufficient.

All changed abstract text must be blue.

### Introduction

Add or revise text to:
- define the practical client-selection problem in DFL,
- explain why topology-aware selection is useful,
- clarify that topology-only selection avoids additional data/gradient exchange,
- acknowledge that it may fail when topology is misaligned with data heterogeneity,
- tone down claims about novelty and scalability.

All changed introduction text must be blue.

### Related Work

Add a stronger related-work discussion covering:
- spectral methods in DFL,
- Laplacian matrix sampling,
- probabilistic communication in heterogeneous edge computing,
- node selection in non-IID FL,
- spectral positional encodings for heterophilous graphs,
- Byzantine-resilient decentralized SGD if used as a baseline or boundary case.

The discussion must clearly position this paper relative to prior Laplacian/spectral approaches.

All changed related-work text must be blue.

### Method

Revise the method section to:
- explicitly define the graph construction assumption,
- define what information is required to compute spectral embeddings and heat-kernel scores,
- add computational complexity,
- explain how neighbor selection is performed after preprocessing,
- justify `k`, `t`, and `beta`,
- state that these parameters are later evaluated in sensitivity analysis.

All changed method text must be blue.

### Experiments

Update the experiments section to include, if feasible:
- stronger baselines,
- parameter ablations,
- larger graphs,
- multiple heterogeneity levels,
- topology/data mismatch experiments,
- dynamic topology stress tests,
- overhead measurements.

All changed experiment text, captions, tables, and result interpretation must be blue.

### Limitations

Add a concise limitations section or paragraph covering:
- dependence on the communication graph,
- possible mismatch between topology and data relevance,
- need for global topology information in spectral preprocessing,
- static-graph assumption,
- possible inadequacy of low-frequency eigenvectors,
- limited scope of topology-only scoring.

All changed limitations text must be blue.

### Conclusion

Revise the conclusion to:
- avoid overstated claims,
- summarize what is empirically supported,
- state that topology-aware client selection is useful but not universally optimal,
- identify hybrid topology/learning-aware selection and dynamic topology handling as future work.

All changed conclusion text must be blue.

## Experimental implementation checklist

Inspect the repository and locate experiment scripts, dataset loaders, configuration files, and plotting utilities. Then implement the following where feasible.

### Baselines

Implement or add configs for:

1. `random_beta`
   - Randomly sample a `beta` fraction of neighbors.

2. `full_neighbor`
   - Aggregate over all neighbors.

3. `degree_or_centrality`
   - Select neighbors using graph degree, PageRank, betweenness, or another simple topology-only baseline.

4. `model_similarity`
   - Select neighbors based on model-update cosine similarity or parameter-distance similarity, if model updates are locally available.

5. `loss_or_validation_based`
   - Select neighbors using local validation/loss signals, if available without violating assumptions.

6. `laplacian_sampling`
   - Include if a faithful implementation is feasible.

7. `node_selection_non_iid_adaptation`
   - Adapt an FL node-selection method to DFL only if the adaptation is technically justified.

### Ablations

Run or configure sweeps for:

```yaml
k: [2, 4, 8, 16]
t: [0.1, 0.5, 1.0, 2.0, 5.0]
beta: [0.25, 0.5, 0.75]
alpha: [0.1, 0.3, 0.5, 1.0]
num_clients: [50, 100, 500, 1000]
seeds: [0, 1, 2]
```

Adjust values only if repository constraints require it. Document any changes.

### Graph families

Add or verify support for:

```yaml
graph_type:
  - erdos_renyi
  - watts_strogatz
  - barabasi_albert
  - random_geometric
  - stochastic_block_model
```

### Topology/data relationship settings

Add settings for:

```yaml
data_topology_alignment:
  - independent
  - topology_correlated
  - heterophilous_mismatch
```

Definitions:

- `independent`: data distribution is assigned independently of graph topology.
- `topology_correlated`: nearby or same-community clients have similar label distributions.
- `heterophilous_mismatch`: nearby or same-community clients have intentionally dissimilar label distributions.

### Dynamic topology

Add settings for:

```yaml
dynamic_topology:
  enabled: true
  edge_dropout_prob: [0.05, 0.1, 0.2]
  node_dropout_prob: [0.05, 0.1]
  rewire_interval: [5, 10]
```

If dynamic topology cannot be fully supported, implement a simple stress test with edge dropout and report it as a limitation.

### Metrics

Record and plot:

```yaml
metrics:
  - test_accuracy_by_round
  - train_loss_by_round
  - communication_bytes_per_round
  - cumulative_communication_bytes
  - rounds_to_target_accuracy
  - selection_time_per_round
  - spectral_preprocessing_time
  - memory_overhead
```

### Output artifacts

Generate or update:

```text
results/
  tables/
    main_results.csv
    ablations_k_t_beta.csv
    scalability.csv
    dynamic_topology.csv
    overhead.csv
  figures/
    accuracy_vs_rounds.pdf
    accuracy_vs_communication.pdf
    ablation_k.pdf
    ablation_t.pdf
    ablation_beta.pdf
    scalability.pdf
    dynamic_topology.pdf
```

Use the repository’s existing naming conventions if they differ.

## Reporting requirements

When updating the manuscript:

1. Do not overclaim.
2. Do not hide negative or mixed results.
3. Report standard deviation or confidence intervals where possible.
4. State the number of seeds.
5. State hardware and runtime environment if overhead results are added.
6. Make the communication budget explicit.
7. Make the graph construction assumptions explicit.
8. Distinguish between topology-only and learning-aware methods.
9. Explain whether spectral preprocessing requires global topology knowledge.
10. Keep the paper within IEEE SPL length constraints. If necessary, move extra details to an appendix, supplementary file, or repository documentation.

## Final repository deliverables

At the end of the work, produce:

1. Updated LaTeX manuscript with all changes in blue.
2. Updated bibliography with the reviewer-suggested references where appropriate.
3. Updated experiment scripts/configs for new baselines and ablations.
4. Generated result tables and plots for any experiments actually run.
5. A `revision_log.md` file with:
   - each reviewer concern,
   - the manuscript section changed,
   - experiments added or not added,
   - reason for any omitted experiment or baseline,
   - list of files modified.
6. A reproducibility note with commands used to run experiments.

## Reviewer concern mapping

Use this mapping to ensure no concern is missed.

| Reviewer concern | Required action |
|---|---|
| Graph construction unclear | Add graph-construction subsection and assumptions |
| Topology may not reflect learning relevance | Add limitations and topology/data mismatch experiment |
| Spectral quantities require global topology | Add preprocessing and decentralization clarification |
| Static graph assumption | Add dynamic topology discussion or experiment |
| Missing gradient/model/data baselines | Add at least one learning-aware baseline or justify omission |
| No convergence or communication trade-off analysis | Add accuracy-vs-rounds and accuracy-vs-communication results |
| No ablations for `k`, `t`, `beta` | Add sweeps and plots/tables |
| Scalability claims unsupported | Add larger graphs and complexity/runtime analysis |
| Novelty overstated | Tone down abstract, introduction, contribution list, and conclusion |
| Low-frequency eigenvectors may be insufficient | Add spectral-design discussion and mixed/full-spectrum ablation if feasible |
| Missing related work | Add reviewer-suggested references with DOI where available |
| Experiments too limited | Add stronger baselines, graph families, heterogeneity levels, and seeds |

## Final self-check before stopping

Before finishing, verify:

- [ ] The manuscript compiles.
- [ ] Every manuscript change is blue in the compiled PDF.
- [ ] No new claim lacks either an experiment, citation, or explicit limitation.
- [ ] The related-work section discusses the reviewer-suggested references.
- [ ] The method no longer claims to be fully decentralized unless justified.
- [ ] Graph construction assumptions are explicit.
- [ ] Complexity and overhead are discussed.
- [ ] Baselines are stronger than random sampling alone.
- [ ] Ablations for `k`, `t`, and `beta` are present or explicitly deferred.
- [ ] Dynamic topology is either experimentally tested or discussed as a limitation.
- [ ] No fabricated results are included.
- [ ] `revision_log.md` exists and is complete.
