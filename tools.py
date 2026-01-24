import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from pylab import mpl
from scipy.optimize import curve_fit

# Specify the default font: Solve the problem of not being able to display Chinese characters.
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

# Solve the problem of the negative sign '-' being displayed as a square when saving images.
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


def GenGraph(data_dir):

    from random import randint

    name_list = os.listdir(data_dir)

    for i in name_list:

        print(i.split(".xlsx")[0])

        name = data_dir + "/" + i.split(".xlsx")[0]
        # res, _, idy, _ = dataRd(name)

        # ★ 接收 result
        res, idx, idy, result = dataRd(name)

        # =========================
        # ① 保存矩阵
        # =========================
        result_save_path = os.path.join(
            "results", f"{name.split('/')[-1]}.xlsx"
        )
        result.to_excel(result_save_path)

        node_label = list(range(0, len(idy)))
        labels = dict(zip(node_label, idy))

        # G = nx.from_numpy_matrix(res)  # python 3.6
        G = nx.from_numpy_array(res)  # python 3.10

        # Rearrange the Graph
        pos = nx.spring_layout(G, iterations=15, seed=2500)

        label_propagation_communities = nx.community.label_propagation_communities(G)

        colors = ["" for x in range(G.number_of_nodes())]  # initialize colors list

        counter = 0  # Number of community

        for com in label_propagation_communities:
            color = "#%06X" % randint(0, 0xFFFFFF)  # Creates random RGB color
            counter += 1
            for node in list(
                com
            ):  # Fill colors list with the particular color for the community nodes
                colors[node] = color

        fig, ax = plt.subplots(figsize=(15, 9))
        nx.draw(
            G, node_size=300, labels=labels, width=2, node_color=colors, pos=pos, ax=ax
        )
        fig.savefig("./figures/{}.png".format(name.split("/")[-1]))
        plt.close()


def GenKeyNodes(data_dir):

    name_list = os.listdir(data_dir)

    for i in name_list:

        print(i.split(".xlsx")[0])

        name = data_dir + "/" + i.split(".xlsx")[0]

        res, _, idy, _ = dataRd(name)

        # G = nx.from_numpy_matrix(res)  # python 3.6
        G = nx.from_numpy_array(res)  # python 3.10

        # Extract the top 15 nodes with the highest eigenvector centrality
        data = dict(nx.eigenvector_centrality(G, max_iter=2000, weight="weight"))

        df = pd.DataFrame(
            {"node": list(data.keys()), "degree": list(data.values()), "label": idy}
        )

        top_30_EC_nodes = df.sort_values(
            by=["degree", "node"], ascending=[False, True]
        ).head(30)

        top_30_EC_nodes.to_csv(
            "./top_30_EC_nodes/top_30_EC_nodes_{}.csv".format(name.split("/")[-1]),
            index=False,
            encoding="utf-8-sig",
        )


def Community(data_dir):

    name_list = os.listdir(data_dir)

    commu_L_total = []
    lp_total = []
    counter_total = []
    bl_tatal = []

    for i in name_list:

        print(i.split(".xlsx")[0])

        name = data_dir + "/" + i.split(".xlsx")[0]

        res, _, _, _ = dataRd(name)

        # G = nx.from_numpy_matrix(res)  # python 3.6
        G = nx.from_numpy_array(res)  # python 3.10

        # Modularity
        commu_L = nx.community.modularity(
            G, nx.community.label_propagation_communities(G), weight="weight"
        )

        print("L-Modularity:", commu_L)

        # Use semi-synchronous label propagation method to detect communities.
        label_propagation_communities = nx.community.label_propagation_communities(G)

        a = list(label_propagation_communities)
        b = sorted(a, key=len)
        lp = []
        for j in range(len(b)):
            lp.append(len(sorted(b[j])))
        print("Community structure:", lp)

        bl = sum(1 for x in lp if x >= 3)
        print("Number of blocks:", bl)

        # Number of community
        counter = len(label_propagation_communities)
        print("Number of community:", counter)

        commu_L_total.append(commu_L)
        lp_total.append(lp)
        counter_total.append(counter)
        bl_tatal.append(bl)

    return commu_L_total, lp_total, counter_total, bl_tatal


def M_cal(a):
    # Monotonicity calculation

    N = len(a)
    b = np.array(a)
    c = b[:, 1]
    count = []
    d = [c[0]]
    for i in c:
        if i not in d:
            d.append(i)
    for j in range(len(d)):
        tmp = 0
        for ii in range(len(c)):
            if d[j] == c[ii]:
                tmp += 1
        count.append(tmp)

    tmp = 0
    for i in range(len(count)):
        tmp += count[i] * (count[i] - 1)
    tmp = tmp / (N * (N - 1))
    M = np.power(1 - tmp, 2)
    return M


def MonCentrality(data_dir):

    name_list = os.listdir(data_dir)

    DC_total = []
    EC_total = []
    CC_total = []
    BC_total = []

    for i in name_list:

        print(i.split(".xlsx")[0])

        name = data_dir + "/" + i.split(".xlsx")[0]
        res, _, _, _ = dataRd(name)

        # G = nx.from_numpy_matrix(res)  # python 3.6
        G = nx.from_numpy_array(res)  # python 3.10

        DC = M_cal(
            sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)
        )
        EC = M_cal(
            sorted(
                nx.eigenvector_centrality(G, max_iter=2000, weight="weight").items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        CC = M_cal(
            sorted(nx.closeness_centrality(G).items(), key=lambda x: x[1], reverse=True)
        )
        BC = M_cal(
            sorted(
                nx.betweenness_centrality(G, weight="weight").items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        print("DC-M:", DC)
        print("EC-M:", EC)
        print("CC-M:", CC)
        print("BC-M:", BC)

        DC_total.append(DC)
        EC_total.append(EC)
        CC_total.append(CC)
        BC_total.append(BC)

    return DC_total, EC_total, CC_total, BC_total

def GenCommunity(data_dir):

    commu_L_total, lp_total, counter_total, bl_total = Community(data_dir)

    name_list = os.listdir(data_dir)

    for k in range(len(name_list)):
        name_list[k] = name_list[k].replace(".xlsx", "")

    commu_L_total = np.array(commu_L_total)
    counter_total = np.array(counter_total)
    bl_total = np.array(bl_total)

    for l in range(len(lp_total)):
        lp_total[l] = str(lp_total[l]).replace("[", "").replace("]", "")
    lp_total = np.array(lp_total)

    result = np.vstack((commu_L_total, counter_total, lp_total, bl_total))

    result = pd.DataFrame(
        result,
        columns=name_list,
        index=["L-Modularity", "Number of community", "Community structure", "Number of blocks"],
    )

    result.to_excel("./results/Community_results.xlsx")

def GenMonCentrality(data_dir):

    DC_total, EC_total, CC_total, BC_total = MonCentrality(data_dir)

    name_list = os.listdir(data_dir)

    DC_total = np.array(DC_total)
    EC_total = np.array(EC_total)
    CC_total = np.array(CC_total)
    BC_total = np.array(BC_total)

    for k in range(len(name_list)):
        name_list[k] = name_list[k].replace(".xlsx", "")

    result = np.vstack((DC_total, EC_total, CC_total, BC_total))

    result = pd.DataFrame(
        result,
        columns=name_list,
        index=["DC", "EC", "CC", "BC"],
    )

    result.to_excel("./results/MonCentrality_results.xlsx")

def TopoStructure(data_dir):

    name_list = os.listdir(data_dir)

    Nodes_total = []
    Edges_total = []
    Diameter_total = []
    Density_total = []
    Assortativity_total = []
    Transitivity_total = []
    AvgDegree_total = []
    AvgShortPathLen_total = []
    AvgCluCoeffi_total = []
    Fd_total = []

    for i in name_list:

        print(i.split(".xlsx")[0])

        name = data_dir + "/" + i.split(".xlsx")[0]
        res, _, _, _ = dataRd(name)

        G = nx.from_numpy_array(res)

        n = G.number_of_nodes()
        m = G.number_of_edges()

        Avg_degree = 2 * m / n
        Density = nx.density(G)
        Transitivity = nx.transitivity(G)
        ACC = nx.average_clustering(G, weight="weight", count_zeros=True)
        Assortativity = nx.degree_assortativity_coefficient(G, weight="weight")

        # 综合难度系数
        Fd = Avg_degree * Density * Transitivity * ACC * 10

        # 最大连通子图
        components = nx.connected_components(G)
        max_component = max(components, key=len)
        mc = list(max_component)
        i0, j0 = np.ix_(mc, mc)
        d_res = res[i0, j0]
        G0 = nx.from_numpy_array(d_res)

        Diameter = nx.diameter(G0)
        ASPL = nx.average_shortest_path_length(G0, weight="weight")

        Nodes_total.append(n)
        Edges_total.append(m)
        Diameter_total.append(Diameter)
        Density_total.append(Density)
        Assortativity_total.append(Assortativity)
        Transitivity_total.append(Transitivity)
        AvgDegree_total.append(Avg_degree)
        AvgShortPathLen_total.append(ASPL)
        AvgCluCoeffi_total.append(ACC)
        Fd_total.append(Fd)

    return (
        Nodes_total,
        Edges_total,
        Diameter_total,
        Density_total,
        Assortativity_total,
        Transitivity_total,
        AvgDegree_total,
        AvgShortPathLen_total,
        AvgCluCoeffi_total,
        Fd_total,
    )

# =====================================================
# Degree distribution + exponent
# =====================================================
def DegreeDistr(data_dir):

    name_list = os.listdir(data_dir)
    DegreeExponent_list = []

    file_path = "./results/DegreeDistr.xlsx"
    if os.path.exists(file_path):
        os.remove(file_path)

    with pd.ExcelWriter(file_path) as writer:

        for ii in name_list:

            print(ii.split(".xlsx")[0])

            name = data_dir + "/" + ii.split(".xlsx")[0]
            res, _, _, _ = dataRd(name)

            G = nx.from_numpy_array(res)

            n = len(G.nodes)
            d = dict(nx.degree(G))
            dd = list(d.values())

            tmp_value = sorted(set(dd))
            tmp_count = [dd.count(v) for v in tmp_value]
            tmp_count_frq = np.array(tmp_count) / n

            x = np.array(tmp_value)
            y = tmp_count_frq

            df = pd.DataFrame({"x": x, "y": y})

            df.to_excel(
                writer,
                sheet_name=ii.split(".xlsx")[0],
                index=None
            )

    return DegreeExponent_list

# =====================================================
# Generate final topological structure table
# =====================================================
def GenTopoStructure(data_dir):

    (
        Nodes_total,
        Edges_total,
        Diameter_total,
        Density_total,
        Assortativity_total,
        Transitivity_total,
        AvgDegree_total,
        AvgShortPathLen_total,
        AvgCluCoeffi_total,
        Fd_total,
    ) = TopoStructure(data_dir)


    name_list = os.listdir(data_dir)
    name_list = [n.replace(".xlsx", "") for n in name_list]

    result = np.vstack(
        (
            Nodes_total,
            Edges_total,
            Diameter_total,
            Density_total,
            Assortativity_total,
            Transitivity_total,
            AvgDegree_total,
            AvgShortPathLen_total,
            AvgCluCoeffi_total,
            Fd_total,
        )
    )

    result = pd.DataFrame(
        result,
        columns=name_list,
        index=[
            "Number of Nodes",
            "Number of Edges",
            "Diameter",
            "Density",
            "Assortativity",
            "Transitivity",
            "Average degree",
            "Average shortest path length",
            "Average clustering coefficient",
            "Comprehensive difficulty coefficient",
        ],
    )

    result.to_excel("./results/Topological_Structure_of_KPNs.xlsx")
