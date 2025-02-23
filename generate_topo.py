import json
import math
import random

def distance(n1, n2):
    return math.sqrt((n1["LocX"] - n2["LocX"])**2 + (n1["LocY"] - n2["LocY"])**2)

def generate_topology(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    
    nodes = data["Nodes"]
    edges = []

    for node in nodes:
        node_id = node["NodeId"]
        distances = [(other["NodeId"], distance(node, other)) for other in nodes if other["NodeId"] != node_id]
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = distances[:4]

        for neighbor_id, _ in nearest_neighbors:
            if not any(edge for edge in edges if edge["SrcNodeID"] == neighbor_id and edge["DstNodeID"] == node_id):
                edges.append({
                    "EdgeType": "Link",
                    "SrcNodeID": node_id,
                    "DstNodeID": neighbor_id,
                    "Bandwidth": random.randint(50, 200)
                })

    data["Edges"] = edges
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

# Example usage
generate_topology("examples/scenarios/configs/trust_config_1.json", "output_topology.json")
