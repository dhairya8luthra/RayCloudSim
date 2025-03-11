import json
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pandas as pd

# Load the JSON file
file_path = "logs\\vis\info4frame.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Store embeddings
embeddings_dict = {}

# Iterate through each snapshot
for snapshot, snapshot_data in data.items():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for node in snapshot_data["node"].keys():
        G.add_node(node)

    # Add edges with weights
    for edge, weight in snapshot_data["edge"].items():
        # Parsing edge tuple
        edge_tuple = eval(edge)  # Convert string representation of tuple to actual tuple
        src, dst, _ = edge_tuple
        G.add_edge(src, dst, weight=weight)

    # Generate node2vec model
    node2vec = Node2Vec(G, dimensions=128, walk_length=10, num_walks=100, workers=4)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # Store embeddings
    embeddings = {node: model.wv[node] for node in G.nodes()}
    embeddings_dict[snapshot] = embeddings

# Convert to DataFrame
all_embeddings = []
for snapshot, emb_dict in embeddings_dict.items():
    for node, emb in emb_dict.items():
        all_embeddings.append([snapshot, node] + emb.tolist())

columns = ["snapshot", "node"] + [f"dim_{i}" for i in range(128)]
df_embeddings = pd.DataFrame(all_embeddings, columns=columns)

# Save embeddings to CSV
df_embeddings.to_csv("node2vec_embeddings.csv", index=False)

print("Node2Vec embeddings have been saved to 'node2vec_embeddings.csv'.")