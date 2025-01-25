import pandas as pd
import networkx as nx
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Hypothesis: Clusters with higher average sentiment scores (confidence) tend to have higher density in the social network.


# 1. Load Data
fixed_dataset = pd.read_json("../../data/fixed_dataset.json")  # Your dataset
edge_df = pd.read_csv("../../data/graph.csv")  # Graph edge list
confidence_df = fixed_dataset[["user_id", "sentiment_score", "sentiment_label"]]

# 2. Build Graph
G = nx.Graph()
for _, row in edge_df.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

# Attach confidence scores as node attributes
confidence_dict = confidence_df.set_index("user_id")["sentiment_score"].to_dict()
nx.set_node_attributes(G, confidence_dict, name="sentiment_score")

# 3. Detect Communities
from networkx.algorithms import community

communities = community.greedy_modularity_communities(G)

# 4. Compute Cluster Metrics
cluster_stats = {"cluster_id": [], "density": [], "avg_confidence": []}

for i, community_nodes in enumerate(communities):
    subgraph = G.subgraph(community_nodes)
    density = nx.density(subgraph)
    avg_confidence = np.mean(
        [G.nodes[node].get("sentiment_score", np.nan) for node in community_nodes]
    )

    cluster_stats["cluster_id"].append(i)
    cluster_stats["density"].append(density)
    cluster_stats["avg_confidence"].append(avg_confidence)

cluster_stats_df = pd.DataFrame(cluster_stats)

# 5. Correlation Analysis
df_clean = cluster_stats_df.dropna()
if len(df_clean) < 2:
    print("Not enough clusters to analyze.")
else:
    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(
        df_clean["density"], df_clean["avg_confidence"]
    )
    print("Pearson Correlation:")
    print(f"  Correlation = {pearson_corr:.4f}")
    print(f"  p-value     = {pearson_p:.4f}")

    # Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(
        df_clean["density"], df_clean["avg_confidence"]
    )
    print("Spearman Correlation:")
    print(f"  Correlation = {spearman_corr:.4f}")
    print(f"  p-value     = {spearman_p:.4f}")

# 6. Visualization
plt.figure(figsize=(10, 6))
plt.scatter(
    df_clean["avg_confidence"], df_clean["density"], alpha=0.7, label="Data Points"
)
m, b = np.polyfit(df_clean["avg_confidence"], df_clean["density"], 1)  # Regression line
plt.plot(
    df_clean["avg_confidence"],
    m * df_clean["avg_confidence"] + b,
    color="red",
    label="Regression Line",
)
plt.xlabel("Average Confidence Score")
plt.ylabel("Cluster Density")
plt.title("Confidence Score vs Cluster Density")
plt.legend()
plt.grid(True)
plt.show()

"""
These results suggest that average sentiment scores are not strongly or significantly correlated with cluster density in this dataset. It may indicate that cluster density is independent of sentiment scores, or there may be other factors affecting density that are not captured by confidence alone.
"""
