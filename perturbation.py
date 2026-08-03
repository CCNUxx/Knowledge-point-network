import os
import numpy as np
import pandas as pd
import networkx as nx
import tools


def random_edge_deletion(G, delete_ratio, rng):
    """Randomly remove a fraction of edges."""

    Gp = G.copy()
    edges = list(Gp.edges())
    if len(edges) == 0:
        return Gp

    n_delete = int(len(edges) * delete_ratio)
    if n_delete == 0:
        return Gp

    delete_idx = rng.choice(len(edges), n_delete, replace=False)
    for i in delete_idx:
        u, v = edges[i]
        if Gp.has_edge(u, v):
            Gp.remove_edge(u, v)
    return Gp


def random_node_deletion(G, delete_ratio, rng):
    """Randomly remove a fraction of nodes."""

    Gp = G.copy()
    nodes = list(Gp.nodes())
    n_delete = int(len(nodes) * delete_ratio)
    if n_delete == 0:
        return Gp

    delete_nodes = rng.choice(nodes, n_delete, replace=False)
    Gp.remove_nodes_from(delete_nodes)
    return Gp


def robustness_analysis(matrix_path, n_repeat=1000, seed=42,
                        edge_range=(0.05, 0.15), node_range=(0.05, 0.15)):
    """Perturbation-based robustness analysis of a KPN matrix.

    Randomly remove 5%-15% of nodes and edges, and recompute the network
    metrics for each realization.
    """

    res, _ = tools.load_matrix(matrix_path)
    G0 = nx.from_numpy_array(res)
    rng = np.random.default_rng(seed)

    # Metrics reported in the paper (main text + appendix)
    metric_names = [
        "Density",
        "Assortativity",
        "Average clustering coefficient",
        "Comprehensive difficulty coefficient (Fd, x10)",
        "Transitivity",
        "Average degree",
        "Average shortest path length",
    ]

    records = []
    for _ in range(n_repeat):
        edge_ratio = rng.uniform(*edge_range)
        node_ratio = rng.uniform(*node_range)

        G = G0.copy()
        G = random_node_deletion(G, node_ratio, rng)
        G = random_edge_deletion(G, edge_ratio, rng)

        metrics = tools.compute_topo_metrics(G)
        selected = {name: metrics[name] for name in metric_names}

        communities = list(nx.community.label_propagation_communities(G))
        selected["Number of communities"] = len(communities)

        records.append(selected)

    return pd.DataFrame(records)


def compute_robust_summary(df):
    """Median / IQR summary of the perturbation results."""

    summary = {}
    for col in df.columns:
        values = df[col].dropna()
        if len(values) == 0:
            summary[col] = {"median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan}
            continue

        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)
        median = values.median()

        summary[col] = {"median": median, "q25": q25, "q75": q75, "iqr": q75 - q25}

    return pd.DataFrame(summary).T


if __name__ == "__main__":
    matrix_path = "data/First stage_matrix.xlsx"
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    df = robustness_analysis(matrix_path, n_repeat=1000, seed=42)
    df.to_csv(os.path.join(save_dir, "perturbation_1000.csv"), index=False)

    summary = compute_robust_summary(df)
    summary.to_csv(os.path.join(save_dir, "perturbation_summary_1000.csv"))
