import pandas as pd
import networkx as nx
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Hypothesis: Users who predominantly post tweets with negative sentiment tend to form denser clusters in the social network.

# -------------------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------------------
print("[Step 1] Loading data...")
fixed_dataset = pd.read_json("../../data/fixed_dataset.json")  # Your dataset
edge_df = pd.read_csv("../../data/graph.csv")  # Graph edge list
confidence_df = fixed_dataset[['user_id', 'sentiment_score', 'sentiment_label']]

# -------------------------------------------------------------------------
# 2. Aggregate Sentiment Data by user_id
# -------------------------------------------------------------------------
print("[Step 2] Aggregating sentiment data by user_id...")
confidence_aggregated = (
    confidence_df.groupby('user_id', as_index=False)
    .agg({'sentiment_score': 'mean', 'sentiment_label': lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan})
)

# Convert to dictionary for graph attributes
confidence_dict = confidence_aggregated.set_index('user_id')[['sentiment_score', 'sentiment_label']].to_dict(
    orient='index')

# -------------------------------------------------------------------------
# 3. Build Graph
# -------------------------------------------------------------------------
print("[Step 3] Building the graph...")
G = nx.Graph()
for _, row in edge_df.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

# Attach sentiment scores and labels as node attributes
nx.set_node_attributes(G, confidence_dict)

# -------------------------------------------------------------------------
# 4. Detect Communities
# -------------------------------------------------------------------------
print("[Step 4] Detecting communities...")
from networkx.algorithms import community

communities = community.greedy_modularity_communities(G)

# -------------------------------------------------------------------------
# 5. Compute Cluster Metrics
# -------------------------------------------------------------------------
print("[Step 5] Computing cluster metrics...")
cluster_stats = {
    'cluster_id': [],
    'density': [],
    'avg_confidence': [],
    'negative_user_ratio': []
}

for i, community_nodes in enumerate(communities):
    subgraph = G.subgraph(community_nodes)
    density = nx.density(subgraph)
    sentiment_scores = [G.nodes[node].get('sentiment_score', np.nan) for node in community_nodes]
    sentiment_labels = [G.nodes[node].get('sentiment_label', None) for node in community_nodes]

    avg_confidence = np.nanmean(sentiment_scores)
    negative_ratio = sum(1 for label in sentiment_labels if label == 'negative') / len(community_nodes)

    cluster_stats['cluster_id'].append(i)
    cluster_stats['density'].append(density)
    cluster_stats['avg_confidence'].append(avg_confidence)
    cluster_stats['negative_user_ratio'].append(negative_ratio)

cluster_stats_df = pd.DataFrame(cluster_stats)

# -------------------------------------------------------------------------
# 6. Analyze Relationship Between Negative Sentiment and Density
# -------------------------------------------------------------------------
print("[Step 6] Analyzing relationship between cluster density and negative user ratio...")
df_clean = cluster_stats_df.dropna()
if len(df_clean) < 2:
    print("Not enough clusters to analyze.")
else:
    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(df_clean['density'], df_clean['negative_user_ratio'])
    print("Pearson Correlation:")
    print(f"  Correlation = {pearson_corr:.4f}")
    print(f"  p-value     = {pearson_p:.4f}")

    # Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(df_clean['density'], df_clean['negative_user_ratio'])
    print("Spearman Correlation:")
    print(f"  Correlation = {spearman_corr:.4f}")
    print(f"  p-value     = {spearman_p:.4f}")

# -------------------------------------------------------------------------
# 7. Visualization
# -------------------------------------------------------------------------
print("[Step 7] Creating visualization...")
output_path = "../plots/negative_user_ratio_vs_cluster_density.png"

plt.figure(figsize=(10, 6))
plt.scatter(df_clean['negative_user_ratio'], df_clean['density'], alpha=0.7, label='Data Points')
m, b = np.polyfit(df_clean['negative_user_ratio'], df_clean['density'], 1)  # Regression line
plt.plot(df_clean['negative_user_ratio'], m * df_clean['negative_user_ratio'] + b, color='red', label='Regression Line')
plt.xlabel('Negative Sentiment Ratio')
plt.ylabel('Cluster Density')
plt.title('Negative Sentiment Ratio vs Cluster Density')
plt.legend()
plt.grid(True)
plt.savefig(output_path)
plt.show()
