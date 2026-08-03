import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
mpl.rcParams["axes.unicode_minus"] = False


def get_res(tmp, res, idx_, idy_):
    for i in range(len(tmp)):
        ttmmpp = tmp.copy()
        del ttmmpp[i]
        tmp_i = np.where(idx_ == tmp[i][1])
        for j in range(len(ttmmpp)):
            tmp_j = np.where(idy_ == ttmmpp[j][1])
            res[tmp_i, tmp_j] = res[tmp_i, tmp_j] + 1


def dataRd(name):
    """Build the co-occurrence matrix from an exam paper (question, knowledge point)."""

    Data = pd.read_excel("./" + name + ".xlsx")
    KnowledgePoints = Data.iloc[:, 0:2]
    KnowledgePoints = np.array(KnowledgePoints)

    jiedian_idx = np.where(KnowledgePoints[:, 1] == "节点")
    KnowledgePoints = np.delete(KnowledgePoints, jiedian_idx, axis=0)
    nan_idx = []
    for i in range(len(KnowledgePoints)):
        if type(KnowledgePoints[:, 1][i]) != str and np.isnan(KnowledgePoints[:, 1][i]):
            nan_idx.append(i)
    KnowledgePoints = np.delete(KnowledgePoints, nan_idx, axis=0)

    for i in range(len(KnowledgePoints) - 1):
        if (
            type(KnowledgePoints[i + 1, 0]) != int
            and type(KnowledgePoints[i + 1, 0]) != str
        ):
            KnowledgePoints[i + 1, 0] = KnowledgePoints[i, 0]
        if type(KnowledgePoints[i, 0]) == str:
            tmp = KnowledgePoints[i, 0][0:2]
            tmp = int(tmp)
            KnowledgePoints[i, 0] = tmp

    idx = []
    idy = []
    idx.append(KnowledgePoints[0, 1])
    idy.append(KnowledgePoints[0, 1])
    for i in range(1, len(KnowledgePoints)):
        if KnowledgePoints[i, 1] not in idx:
            idx.append(KnowledgePoints[i, 1])
        if KnowledgePoints[i, 1] not in idy:
            idy.append(KnowledgePoints[i, 1])

    res_ = []
    i = 0
    while i < len(KnowledgePoints):
        tmp = []
        tmp.append(KnowledgePoints[i])
        if i >= len(KnowledgePoints) - 1:
            res_.append(tmp)
            break
        j = 1
        while KnowledgePoints[i, 0] == KnowledgePoints[i + j, 0]:
            tmp.append(KnowledgePoints[i + j])
            j = j + 1
            if i + j > len(KnowledgePoints) - 1:
                break
        res_.append(tmp)
        i = i + j

    res = np.zeros((len(idx), len(idy)))
    idx_ = np.array(idx)
    idy_ = np.array(idy)

    for p in res_:
        get_res(p, res, idx_, idy_)

    result = pd.DataFrame(res, columns=idy, index=idx)

    return res, idx, idy, result


def load_matrix(path):
    """Load a co-occurrence adjacency matrix from an xlsx file."""

    df = pd.read_excel(path, index_col=0)
    return df.values.astype(float), list(df.index)


def compute_topo_metrics(G):
    """Topological metrics of a KPN."""

    n = G.number_of_nodes()
    m = G.number_of_edges()

    avg_degree = 2 * m / n
    density = nx.density(G)
    transitivity = nx.transitivity(G)
    avg_clustering = nx.average_clustering(G, weight="weight", count_zeros=True)
    assortativity = nx.degree_assortativity_coefficient(G, weight="weight")
    fd = avg_degree * density * transitivity * avg_clustering * 10

    try:
        sub = G.subgraph(max(nx.connected_components(G), key=len))
        diameter = nx.diameter(sub)
        aspl = nx.average_shortest_path_length(sub, weight="weight")
    except Exception:
        diameter = np.nan
        aspl = np.nan

    return {
        "Number of Nodes": n,
        "Number of Edges": m,
        "Diameter": diameter,
        "Density": density,
        "Assortativity": assortativity,
        "Transitivity": transitivity,
        "Average degree": avg_degree,
        "Average shortest path length": aspl,
        "Average clustering coefficient": avg_clustering,
        "Comprehensive difficulty coefficient (Fd, x10)": fd,
    }


def top_EC_nodes(G, top_n=15):
    """Top-N knowledge points by eigenvector centrality."""

    ec = nx.eigenvector_centrality(G, max_iter=2000, weight="weight")
    return sorted(ec.items(), key=lambda x: (-x[1], x[0]))[:top_n]


def community_metrics(G):
    """Number of communities and number of blocks.

    A block excludes isolated nodes, node pairs and non-closed triangles;
    a 3-node community counts as a block only if it is a closed triangle.
    """

    communities = list(nx.community.label_propagation_communities(G))

    num_blocks = 0
    for c in communities:
        if len(c) > 3:
            num_blocks += 1
        elif len(c) == 3 and G.subgraph(c).number_of_edges() == 3:
            num_blocks += 1

    return len(communities), num_blocks


def draw_network(G, labels, save_path):
    """Draw a KPN with nodes colored by community."""

    from random import randint

    pos = nx.spring_layout(G, iterations=15, seed=2500)

    label_propagation_communities = nx.community.label_propagation_communities(G)

    colors = ["" for x in range(G.number_of_nodes())]

    for com in label_propagation_communities:
        color = "#%06X" % randint(0, 0xFFFFFF)
        for node in list(com):
            colors[node] = color

    fig, ax = plt.subplots(figsize=(15, 9))
    nx.draw(
        G, node_size=300, labels=labels, width=2, node_color=colors, pos=pos, ax=ax
    )
    fig.savefig(save_path)
    plt.close()
