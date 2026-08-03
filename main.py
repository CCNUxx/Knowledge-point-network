import os
import pandas as pd
import networkx as nx
import tools
import perturbation

results_path = "./results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

figures_path = "./figures"
if not os.path.exists(figures_path):
    os.makedirs(figures_path)

# =====================================================
# 1. Build the KPNs (exam papers -> co-occurrence matrix)
# =====================================================
graphs = {}
labels = {}

for name in ["13V2", "17V1"]:
    res, _, idy, _ = tools.dataRd("data/" + name)
    graphs[name] = nx.from_numpy_array(res)
    labels[name] = idy
    pd.DataFrame(res, index=idy, columns=idy).to_excel(f"./results/{name}_matrix.xlsx")

res_fs, nodes_fs = tools.load_matrix("data/First stage_matrix.xlsx")
graphs["First stage"] = nx.from_numpy_array(res_fs)
labels["First stage"] = nodes_fs

# =====================================================
# 2. Draw all three networks
# =====================================================
for name, G in graphs.items():
    node_labels = {i: labels[name][i] for i in range(len(labels[name]))}
    tools.draw_network(G, node_labels, f"./figures/{name}.png")

# =====================================================
# 3. Topological metrics and community metrics
# =====================================================
metrics = {name: tools.compute_topo_metrics(G) for name, G in graphs.items()}

for name, G in graphs.items():
    num_communities, num_blocks = tools.community_metrics(G)
    metrics[name]["Number of communities"] = num_communities
    metrics[name]["Number of blocks"] = num_blocks

result = pd.DataFrame(metrics)
result.to_excel("./results/Topological_Structure.xlsx")
print(result.to_string())

# =====================================================
# 4. Top-15 knowledge points by eigenvector centrality
# =====================================================
ec_rows = {}
for name, G in graphs.items():
    ec = tools.top_EC_nodes(G)
    ec_rows[name] = [labels[name][n] for n, _ in ec]
    ec_rows[name + "_EC"] = [v for _, v in ec]

df_ec = pd.DataFrame(ec_rows)
df_ec.to_csv("./results/Top15_EC_nodes.csv", index=False, encoding="utf-8-sig")
print(df_ec.to_string())

# =====================================================
# 5. Perturbation-based robustness analysis (First stage)
# =====================================================
print("\nRunning perturbation analysis...")
df_robust = perturbation.robustness_analysis("data/First stage_matrix.xlsx", n_repeat=1000, seed=42)
df_robust.to_csv("./results/perturbation_1000.csv", index=False)

summary = perturbation.compute_robust_summary(df_robust)
summary.to_csv("./results/perturbation_summary_1000.csv")
print(summary.to_string())

print("end")
